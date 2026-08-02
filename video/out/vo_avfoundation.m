/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Direct scanout VO for macOS: hands VideoToolbox-decoded CVPixelBuffers to
 * an AVSampleBufferDisplayLayer. macOS performs HDR/EDR tone mapping natively
 * (same display pipeline as QuickTime), with no GPU shader passes in mpv.
 */

#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <libavutil/hwcontext.h>

#include "common/common.h"
#include "osdep/mac/swift.h"
#include "osdep/timer.h"
#include "sub/osd.h"
#include "video/hwdec.h"
#include "video/mp_image.h"
#include "vo.h"

struct priv {
    AVFCommon *mac;
    struct mp_hwdec_ctx hwctx;
    struct mp_image *next_image;
    CMVideoFormatDescriptionRef format_desc;

    // OSD/subtitle overlay state
    double osd_pts;
    struct mp_osd_res osd_res;
    int64_t osd_change_id;
    bool osd_empty;
    int64_t osd_last_ns;

    // upload pool for software-decoded frames (formats VideoToolbox
    // can't hardware-decode still display through the native layer)
    CVPixelBufferPoolRef upload_pool;
    int upload_w, upload_h;
    OSType upload_fmt;
    // per-item double-buffered IOSurface atlases; parts are GPU-composited
    // CALayers cropping into these via contentsRect
    struct {
        CVPixelBufferRef pb[2];
        int front, w, h, change_id;
    } osd_items[MAX_OSD_PARTS];
};

#define OSD_MAX_LAYERS 512

// Attach colorimetry from mpv's params if the decoder didn't tag the buffer;
// macOS needs these to select the correct (HDR) tone mapping.
static void set_color_attachments(struct vo *vo, CVPixelBufferRef pixbuf)
{
    struct mp_image_params *params = vo->params;
    if (!params)
        return;

    if (!CVBufferGetAttachment(pixbuf, kCVImageBufferColorPrimariesKey, NULL)) {
        CFStringRef prim = NULL;
        switch (params->color.primaries) {
        case PL_COLOR_PRIM_BT_2020: prim = kCVImageBufferColorPrimaries_ITU_R_2020; break;
        case PL_COLOR_PRIM_BT_709:  prim = kCVImageBufferColorPrimaries_ITU_R_709_2; break;
        default: break;
        }
        if (prim)
            CVBufferSetAttachment(pixbuf, kCVImageBufferColorPrimariesKey, prim,
                                  kCVAttachmentMode_ShouldPropagate);
    }

    if (!CVBufferGetAttachment(pixbuf, kCVImageBufferTransferFunctionKey, NULL)) {
        CFStringRef trc = NULL;
        switch (params->color.transfer) {
        case PL_COLOR_TRC_PQ:  trc = kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ; break;
        case PL_COLOR_TRC_HLG: trc = kCVImageBufferTransferFunction_ITU_R_2100_HLG; break;
        default: break;
        }
        if (trc)
            CVBufferSetAttachment(pixbuf, kCVImageBufferTransferFunctionKey, trc,
                                  kCVAttachmentMode_ShouldPropagate);
    }

    if (!CVBufferGetAttachment(pixbuf, kCVImageBufferYCbCrMatrixKey, NULL)) {
        CFStringRef mtx = NULL;
        switch (params->repr.sys) {
        case PL_COLOR_SYSTEM_BT_601:     mtx = kCVImageBufferYCbCrMatrix_ITU_R_601_4; break;
        case PL_COLOR_SYSTEM_BT_709:     mtx = kCVImageBufferYCbCrMatrix_ITU_R_709_2; break;
        case PL_COLOR_SYSTEM_BT_2020_NC: mtx = kCVImageBufferYCbCrMatrix_ITU_R_2020; break;
        default: break;
        }
        if (mtx)
            CVBufferSetAttachment(pixbuf, kCVImageBufferYCbCrMatrixKey, mtx,
                                  kCVAttachmentMode_ShouldPropagate);
    }
}

// copy a software NV12/P010 frame into an IOSurface-backed CVPixelBuffer
static CVPixelBufferRef upload_sw_frame(struct vo *vo, struct mp_image *mpi)
{
    struct priv *p = vo->priv;

    bool p010 = mpi->imgfmt == IMGFMT_P010;
    bool full = vo->params && vo->params->repr.levels == PL_COLOR_LEVELS_FULL;
    OSType fmt = p010
        ? (full ? kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
                : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)
        : (full ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);

    if (!p->upload_pool || p->upload_w != mpi->w || p->upload_h != mpi->h ||
        p->upload_fmt != fmt)
    {
        if (p->upload_pool) {
            CVPixelBufferPoolRelease(p->upload_pool);
            p->upload_pool = NULL;
        }
        NSDictionary *attrs = @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(fmt),
            (id)kCVPixelBufferWidthKey: @(mpi->w),
            (id)kCVPixelBufferHeightKey: @(mpi->h),
            (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
        };
        if (CVPixelBufferPoolCreate(kCFAllocatorDefault, NULL,
                (__bridge CFDictionaryRef)attrs,
                &p->upload_pool) != kCVReturnSuccess)
            return NULL;
        p->upload_w = mpi->w;
        p->upload_h = mpi->h;
        p->upload_fmt = fmt;
    }

    CVPixelBufferRef pb = NULL;
    if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, p->upload_pool,
                                           &pb) != kCVReturnSuccess || !pb)
        return NULL;

    CVPixelBufferLockBaseAddress(pb, 0);
    for (int i = 0; i < 2; i++) {
        uint8_t *dst = CVPixelBufferGetBaseAddressOfPlane(pb, i);
        size_t dst_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, i);
        int rows = CVPixelBufferGetHeightOfPlane(pb, i);
        size_t bytes = (size_t)mpi->w * (p010 ? 2 : 1);
        for (int y = 0; y < rows; y++)
            memcpy(dst + (size_t)y * dst_stride,
                   mpi->planes[i] + (int64_t)y * mpi->stride[i], bytes);
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return pb;  // +1
}

static void clear_osd(struct priv *p)
{
    if (!p->osd_empty) {
        [p->mac setOsdParts:NULL surfaceCount:0 descs:NULL partCount:0];
        p->osd_empty = true;
    }
}

// upload one item's packed BGRA atlas into its back-buffer IOSurface;
// returns the buffer to display, or NULL on failure
static CVPixelBufferRef upload_osd_item(struct priv *p, struct sub_bitmaps *imgs)
{
    int slot = imgs->render_index;
    if (slot < 0 || slot >= MAX_OSD_PARTS)
        return NULL;
    __typeof__(&p->osd_items[0]) it = &p->osd_items[slot];

    int pw = imgs->packed_w, ph = imgs->packed_h;
    bool resize = it->w != pw || it->h != ph;
    if (!resize && it->pb[it->front] && it->change_id == imgs->change_id)
        return it->pb[it->front];  // unchanged (e.g. static subtitles)

    int back = it->front ^ 1;
    if (resize || !it->pb[back]) {
        if (resize) {
            for (int i = 0; i < 2; i++) {
                if (it->pb[i]) {
                    CVPixelBufferRelease(it->pb[i]);
                    it->pb[i] = NULL;
                }
            }
        }
        NSDictionary *attrs = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
        if (CVPixelBufferCreate(kCFAllocatorDefault, pw, ph,
                kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attrs,
                &it->pb[back]) != kCVReturnSuccess) {
            it->pb[back] = NULL;
            return NULL;
        }
        it->w = pw;
        it->h = ph;
    }

    CVPixelBufferRef pb = it->pb[back];
    CVPixelBufferLockBaseAddress(pb, 0);
    uint8_t *dst = CVPixelBufferGetBaseAddress(pb);
    size_t dst_stride = CVPixelBufferGetBytesPerRow(pb);
    const uint8_t *src = imgs->packed->planes[0];
    size_t src_stride = imgs->packed->stride[0];
    for (int y = 0; y < ph; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, (size_t)pw * 4);
    CVPixelBufferUnlockBaseAddress(pb, 0);

    it->front = back;
    it->change_id = imgs->change_id;
    return pb;
}

static void update_osd(struct vo *vo)
{
    struct priv *p = vo->priv;

    // cap the core-side ASS rasterization/packing rate; layer updates
    // themselves are cheap (GPU-composited)
    int64_t now = mp_time_ns();
    if (now - p->osd_last_ns < MP_TIME_MS_TO_NS(16))
        return;
    p->osd_last_ns = now;

    CGSize sz = [p->mac window].framePixel.size;
    int W = sz.width, H = sz.height;
    if (W < 1 || H < 1)
        return;

    // margins = the letterbox borders around the aspect-fitted video
    struct mp_osd_res res = { .w = W, .h = H, .display_par = 1 };
    if (vo->params) {
        int dw, dh;
        mp_image_params_get_dsize(vo->params, &dw, &dh);
        if (dw > 0 && dh > 0) {
            int fw = W, fh = H;
            if ((int64_t)W * dh > (int64_t)H * dw)
                fw = (int64_t)H * dw / dh;
            else
                fh = (int64_t)W * dh / dw;
            res.ml = res.mr = (W - fw) / 2;
            res.mt = res.mb = (H - fh) / 2;
        }
    }

    static const bool formats[SUBBITMAP_COUNT] = {
        [SUBBITMAP_BGRA] = true,
    };
    struct sub_bitmap_list *list =
        osd_render(vo->osd, res, p->osd_pts, 0, formats);

    if (list->change_id == p->osd_change_id && osd_res_equals(res, p->osd_res))
        goto done;
    p->osd_change_id = list->change_id;
    p->osd_res = res;

    // upload each item's atlas (skipped when unchanged) and describe every
    // part as {surface, src rect, dest rect} for GPU-composited CALayers
    void *surfaces[MAX_OSD_PARTS] = {0};
    int32_t descs[OSD_MAX_LAYERS * 9];
    int ns = 0, np = 0;

    for (int i = 0; i < list->num_items; i++) {
        struct sub_bitmaps *imgs = list->items[i];
        if (!imgs->num_parts || !imgs->packed)
            continue;
        CVPixelBufferRef pb = upload_osd_item(p, imgs);
        if (!pb)
            continue;
        IOSurfaceRef surf = CVPixelBufferGetIOSurface(pb);
        if (!surf)
            continue;
        surfaces[ns] = surf;

        for (int j = 0; j < imgs->num_parts && np < OSD_MAX_LAYERS; j++) {
            struct sub_bitmap *sb = &imgs->parts[j];
            int32_t *d = &descs[np * 9];
            d[0] = ns;
            d[1] = sb->src_x;
            d[2] = sb->src_y;
            d[3] = sb->w;
            d[4] = sb->h;
            d[5] = sb->x;
            d[6] = sb->y;
            d[7] = sb->dw ? sb->dw : sb->w;
            d[8] = sb->dh ? sb->dh : sb->h;
            np++;
        }
        ns++;
    }

    if (!np) {
        clear_osd(p);
        goto done;
    }

    [p->mac setOsdParts:surfaces surfaceCount:ns descs:descs partCount:np];
    p->osd_empty = false;

done:
    talloc_free(list);
}

static int preinit(struct vo *vo)
{
    struct priv *p = vo->priv;

    if (!NSApp) {
        MP_ERR(vo, "No NSApplication initialized, vo_avfoundation requires "
                   "the mpv binary\n");
        return -1;
    }

    p->mac = [[AVFCommon alloc] init:vo];
    if (!p->mac)
        return -1;

    vo->hwdec_devs = hwdec_devices_create();
    p->hwctx = (struct mp_hwdec_ctx){
        .driver_name = "avfoundation",
        .hw_imgfmt = IMGFMT_VIDEOTOOLBOX,
    };

    int ret = av_hwdevice_ctx_create(&p->hwctx.av_device_ref,
                                     AV_HWDEVICE_TYPE_VIDEOTOOLBOX, NULL, NULL, 0);
    if (ret != 0) {
        MP_ERR(vo, "Failed to create VideoToolbox hwdevice_ctx\n");
        return -1;
    }

    hwdec_devices_add(vo->hwdec_devs, &p->hwctx);
    return 0;
}

static int query_format(struct vo *vo, int format)
{
    // hw frames pass through; sw frames are uploaded (mpv auto-converts
    // other sw formats to one of these)
    return format == IMGFMT_VIDEOTOOLBOX || format == IMGFMT_NV12 ||
           format == IMGFMT_P010;
}

static int reconfig(struct vo *vo, struct mp_image_params *params)
{
    struct priv *p = vo->priv;

    bool hdr = params && (params->color.transfer == PL_COLOR_TRC_PQ ||
                          params->color.transfer == PL_COLOR_TRC_HLG);
    [p->mac setWantsEdr:hdr];

    MP_VERBOSE(vo, "reconfig flush (hdr=%d)\n", hdr);
    [p->mac flush];
    if (p->format_desc) {
        CFRelease(p->format_desc);
        p->format_desc = NULL;
    }

    if (![p->mac config:vo])
        return -1;
    return 0;
}

static bool draw_frame(struct vo *vo, struct vo_frame *frame)
{
    struct priv *p = vo->priv;

    mp_image_t *mpi = NULL;
    if (!frame->redraw && !frame->repeat)
        mpi = mp_image_new_ref(frame->current);
    if (mpi)
        p->osd_pts = mpi->pts;

    talloc_free(p->next_image);
    p->next_image = mpi;

    update_osd(vo);
    return VO_TRUE;
}

static void flip_page(struct vo *vo)
{
    struct priv *p = vo->priv;
    if (!p->next_image)
        return;

    CVPixelBufferRef pixbuf = NULL;
    bool owned = false;
    if (p->next_image->imgfmt == IMGFMT_VIDEOTOOLBOX) {
        pixbuf = (CVPixelBufferRef)p->next_image->planes[3];
    } else {
        pixbuf = upload_sw_frame(vo, p->next_image);
        owned = true;
    }
    if (!pixbuf)
        goto done;

    set_color_attachments(vo, pixbuf);

    if (!p->format_desc ||
        !CMVideoFormatDescriptionMatchesImageBuffer(p->format_desc, pixbuf))
    {
        if (p->format_desc)
            CFRelease(p->format_desc);
        p->format_desc = NULL;
        if (CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault,
                pixbuf, &p->format_desc) != noErr)
        {
            MP_ERR(vo, "Failed to create video format description\n");
            goto done;
        }
    }

    // mpv already called this at display time; present immediately
    CMSampleTimingInfo timing = {
        .duration = kCMTimeInvalid,
        .presentationTimeStamp = kCMTimeInvalid,
        .decodeTimeStamp = kCMTimeInvalid,
    };
    CMSampleBufferRef sbuf = NULL;
    if (CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault, pixbuf,
            p->format_desc, &timing, &sbuf) != noErr || !sbuf)
    {
        MP_ERR(vo, "Failed to create sample buffer\n");
        goto done;
    }

    CFArrayRef attachments = CMSampleBufferGetSampleAttachmentsArray(sbuf, true);
    CFMutableDictionaryRef dict =
        (CFMutableDictionaryRef)CFArrayGetValueAtIndex(attachments, 0);
    CFDictionarySetValue(dict, kCMSampleAttachmentKey_DisplayImmediately,
                         kCFBooleanTrue);

    [p->mac enqueue:(void *)sbuf];
    CFRelease(sbuf);

done:
    if (owned && pixbuf)
        CVPixelBufferRelease(pixbuf);
    mp_image_unrefp(&p->next_image);
    // block until the next display refresh so display-sync sees real vsyncs
    [p->mac swapBuffer];
}

static void get_vsync(struct vo *vo, struct vo_vsync_info *info)
{
    struct priv *p = vo->priv;
    [p->mac fillVsyncWithInfo:info];
}

static int control(struct vo *vo, uint32_t request, void *data)
{
    struct priv *p = vo->priv;

    int events = 0;
    int ret = [p->mac control:vo events:&events request:request data:data];
    // resizing is handled by the layer's videoGravity; just forward events
    vo_event(vo, events);
    return ret;
}

static void uninit(struct vo *vo)
{
    struct priv *p = vo->priv;

    mp_image_unrefp(&p->next_image);
    if (p->format_desc)
        CFRelease(p->format_desc);
    for (int i = 0; i < MAX_OSD_PARTS; i++) {
        for (int j = 0; j < 2; j++) {
            if (p->osd_items[i].pb[j])
                CVPixelBufferRelease(p->osd_items[i].pb[j]);
        }
    }
    if (p->upload_pool)
        CVPixelBufferPoolRelease(p->upload_pool);

    hwdec_devices_remove(vo->hwdec_devs, &p->hwctx);
    av_buffer_unref(&p->hwctx.av_device_ref);

    [p->mac uninit:vo];
    p->mac = nil;
}

const struct vo_driver video_out_avfoundation = {
    .description = "AVFoundation (macOS native scanout, OS-side HDR)",
    .name = "avfoundation",
    .preinit = preinit,
    .query_format = query_format,
    .reconfig = reconfig,
    .control = control,
    .draw_frame = draw_frame,
    .flip_page = flip_page,
    .get_vsync = get_vsync,
    .uninit = uninit,
    .priv_size = sizeof(struct priv),
};
