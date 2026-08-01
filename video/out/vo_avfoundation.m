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
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <libavutil/hwcontext.h>

#include "common/common.h"
#include "osdep/mac/swift.h"
#include "video/hwdec.h"
#include "video/mp_image.h"
#include "vo.h"

struct priv {
    AVFCommon *mac;
    struct mp_hwdec_ctx hwctx;
    struct mp_image *next_image;
    CMVideoFormatDescriptionRef format_desc;
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

    talloc_free(p->next_image);
    p->next_image = mpi;
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

    hwdec_devices_remove(vo->hwdec_devs, &p->hwctx);
    av_buffer_unref(&p->hwctx.av_device_ref);

    [p->mac uninit:vo];
    p->mac = nil;
}

const struct vo_driver video_out_avfoundation = {
    .description = "AVFoundation (macOS native scanout, OS-side HDR)",
    .name = "avfoundation",
    .caps = VO_CAP_NORETAIN,
    .preinit = preinit,
    .query_format = query_format,
    .reconfig = reconfig,
    .control = control,
    .draw_frame = draw_frame,
    .flip_page = flip_page,
    .uninit = uninit,
    .priv_size = sizeof(struct priv),
};
