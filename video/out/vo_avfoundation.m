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
    // ping-pong IOSurface buffers; each remembers which rows it dirtied so
    // the next reuse clears only those instead of the whole 4K surface
    CVPixelBufferRef osd_pb[2];
    int osd_pb_dirty[2][2];
    int osd_pb_idx;
    int osd_pb_w, osd_pb_h;
};

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
}

// src-over blend of a premultiplied BGRA part into the bounding-box buffer
// (bx,by = buffer origin in window coords), with nearest scaling (w,h -> dw,dh).
// Returns true if any non-transparent pixel was written.
static bool blend_part(uint32_t *buf, int bx, int by, int W, int H,
                       struct sub_bitmap *sb)
{
    int dw = sb->dw ? sb->dw : sb->w;
    int dh = sb->dh ? sb->dh : sb->h;
    if (dw <= 0 || dh <= 0)
        return false;

    bool scaled = dw != sb->w || dh != sb->h;
    bool visible = false;
    for (int y = 0; y < dh; y++) {
        int dy = sb->y + y - by;
        if (dy < 0 || dy >= H)
            continue;
        const uint32_t *src_row =
            (const uint32_t *)((const uint8_t *)sb->bitmap +
                               (int64_t)(scaled ? y * sb->h / dh : y) * sb->stride);
        uint32_t *dst_row = buf + (int64_t)dy * W;
        for (int x = 0; x < dw; x++) {
            int dx = sb->x + x - bx;
            if (dx < 0 || dx >= W)
                continue;
            uint32_t s = src_row[scaled ? x * sb->w / dw : x];
            if (!s)
                continue;
            visible = true;
            uint32_t sa = s >> 24;
            if (sa == 255 || !dst_row[dx]) {
                dst_row[dx] = s;
            } else {
                uint32_t d = dst_row[dx], inv = 255 - sa;
                uint32_t b = (s & 0xff) + ((d & 0xff) * inv + 127) / 255;
                uint32_t g = ((s >> 8) & 0xff) + (((d >> 8) & 0xff) * inv + 127) / 255;
                uint32_t r = ((s >> 16) & 0xff) + (((d >> 16) & 0xff) * inv + 127) / 255;
                uint32_t a = sa + ((d >> 24) * inv + 127) / 255;
                dst_row[dx] = (MPMIN(a, 255u) << 24) | (MPMIN(r, 255u) << 16) |
                              (MPMIN(g, 255u) << 8) | MPMIN(b, 255u);
            }
        }
    }
    return visible;
}

static void clear_osd(struct priv *p)
{
    if (!p->osd_empty) {
        [p->mac setOsd:NULL];
        p->osd_empty = true;
    }
}

// IOSurface-backed BGRA buffers: CoreAnimation maps them zero-copy, so even
// per-tick OSC updates never cost the main thread a full-window image copy
static bool ensure_osd_buffers(struct priv *p, int W, int H)
{
    if (p->osd_pb[0] && p->osd_pb_w == W && p->osd_pb_h == H)
        return true;
    for (int i = 0; i < 2; i++) {
        if (p->osd_pb[i]) {
            CVPixelBufferRelease(p->osd_pb[i]);
            p->osd_pb[i] = NULL;
        }
    }

    NSDictionary *attrs = @{
        (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
    };
    for (int i = 0; i < 2; i++) {
        if (CVPixelBufferCreate(kCFAllocatorDefault, W, H,
                kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attrs,
                &p->osd_pb[i]) != kCVReturnSuccess)
            return false;
        p->osd_pb_dirty[i][0] = 0;
        p->osd_pb_dirty[i][1] = H;  // fresh memory: clear everything once
    }
    p->osd_pb_w = W;
    p->osd_pb_h = H;
    return true;
}

static void update_osd(struct vo *vo)
{
    struct priv *p = vo->priv;

    // scripted OSCs animate every tick; rendering+packing a 4K ASS overlay
    // per video frame costs a full core, so cap OSD updates at 30 Hz
    int64_t now = mp_time_ns();
    if (now - p->osd_last_ns < MP_TIME_MS_TO_NS(33))
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

    // row range covered by the current parts (for the next reuse's clear)
    int cur_y0 = H, cur_y1 = 0, num_parts = 0;
    for (int i = 0; i < list->num_items; i++) {
        struct sub_bitmaps *imgs = list->items[i];
        for (int j = 0; j < imgs->num_parts; j++) {
            struct sub_bitmap *sb = &imgs->parts[j];
            int dh = sb->dh ? sb->dh : sb->h;
            cur_y0 = MPMIN(cur_y0, MPMAX(sb->y, 0));
            cur_y1 = MPMAX(cur_y1, MPMIN(sb->y + dh, H));
            num_parts++;
        }
    }
    if (!num_parts || cur_y1 <= cur_y0) {
        clear_osd(p);
        goto done;
    }

    if (!ensure_osd_buffers(p, W, H)) {
        clear_osd(p);
        goto done;
    }

    int idx = p->osd_pb_idx ^= 1;
    CVPixelBufferRef pb = p->osd_pb[idx];

    CVPixelBufferLockBaseAddress(pb, 0);
    uint32_t *buf = CVPixelBufferGetBaseAddress(pb);
    size_t bpr = CVPixelBufferGetBytesPerRow(pb);
    int stride32 = bpr / 4;

    // clear only the rows this buffer dirtied last time it was displayed
    int c0 = p->osd_pb_dirty[idx][0], c1 = p->osd_pb_dirty[idx][1];
    if (c1 > c0)
        memset((uint8_t *)buf + (size_t)c0 * bpr, 0, (size_t)(c1 - c0) * bpr);

    bool visible = false;
    for (int i = 0; i < list->num_items; i++) {
        struct sub_bitmaps *imgs = list->items[i];
        for (int j = 0; j < imgs->num_parts; j++)
            visible |= blend_part(buf, 0, 0, stride32, H, &imgs->parts[j]);
    }
    p->osd_pb_dirty[idx][0] = cur_y0;
    p->osd_pb_dirty[idx][1] = cur_y1;
    CVPixelBufferUnlockBaseAddress(pb, 0);

    // e.g. a faded-out OSC still submits fully transparent bitmaps every tick
    if (!visible) {
        clear_osd(p);
        goto done;
    }

    CVPixelBufferRetain(pb);
    [p->mac setOsd:(void *)pb];  // transfers the +1 retain
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
    return format == IMGFMT_VIDEOTOOLBOX;
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

    CVPixelBufferRef pixbuf = (CVPixelBufferRef)p->next_image->planes[3];
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
    for (int i = 0; i < 2; i++) {
        if (p->osd_pb[i])
            CVPixelBufferRelease(p->osd_pb[i]);
    }

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
