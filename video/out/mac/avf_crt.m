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

#import <Metal/Metal.h>

#include "avf_crt.h"
#include "common/msg.h"
#include "video/mp_image.h"

// Faithful Metal port of shaders/mpv-retro-shaders/crt-lottes.glsl (public
// domain Timothy Lottes CRT; the mpv port's horizontal taps collapse to the
// center sample, which is preserved here). YCbCr->RGB is folded in since the
// scanout path has no other place to do it.
static NSString *const kernel_src = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
constant float HARD_SCAN = -8.0;\n\
constant float MASK_DARK = 0.5;\n\
constant float MASK_LIGHT = 1.5;\n\
constant float BRIGHTNESS_BOOST = 1.0;\n\
constant float HARD_BLOOM_SCAN = -2.0;\n\
constant float BLOOM_AMOUNT = 1.0/16.0;\n\
constant float SHAPE = 2.0;\n\
struct Prm {\n\
    float2 videoSize;\n\
    float2 outSize;\n\
    float lumaScale;\n\
    int mtx601;\n\
};\n\
static float3 linearize3(float3 c) {\n\
    c = max(c, 0.0);\n\
    return 0.87031054496765136718 * pow(c + 0.05958483740687370300, 2.4);\n\
}\n\
static float3 delinearize3(float3 c) {\n\
    c = max(c, 0.0);\n\
    return pow(1.14901518821716308593 * c, 1.0/2.4) - 0.05958483740687370300;\n\
}\n\
static float gauss1d(float p, float scale) {\n\
    return exp2(scale * pow(abs(p), SHAPE));\n\
}\n\
kernel void crt_lottes(texture2d<float, access::sample> luma [[texture(0)]],\n\
                       texture2d<float, access::sample> chroma [[texture(1)]],\n\
                       texture2d<float, access::write> outTex [[texture(2)]],\n\
                       constant Prm &prm [[buffer(0)]],\n\
                       uint2 gid [[thread_position_in_grid]])\n\
{\n\
    if (gid.x >= uint(prm.outSize.x) || gid.y >= uint(prm.outSize.y)) return;\n\
    constexpr sampler smp(filter::linear, address::clamp_to_edge);\n\
    float2 pos = (float2(gid) + 0.5) / prm.outSize;\n\
    float y = luma.sample(smp, pos).r * prm.lumaScale;\n\
    float2 c = chroma.sample(smp, pos).rg * prm.lumaScale;\n\
    y = (y - 16.0/255.0) / (219.0/255.0);\n\
    c = (c - 128.0/255.0) / (224.0/255.0);\n\
    float3 rgb;\n\
    if (prm.mtx601 != 0)\n\
        rgb = float3(y + 1.402*c.y, y - 0.344136*c.x - 0.714136*c.y, y + 1.772*c.x);\n\
    else\n\
        rgb = float3(y + 1.5748*c.y, y - 0.18732*c.x - 0.46812*c.y, y + 1.8556*c.x);\n\
    float3 lin = linearize3(BRIGHTNESS_BOOST * clamp(rgb, 0.0, 1.0));\n\
    float dsty = -fract(pos.y * prm.videoSize.y - 0.5);\n\
    float wsum = gauss1d(dsty - 1.0, HARD_SCAN) + gauss1d(dsty, HARD_SCAN) +\n\
                 gauss1d(dsty + 1.0, HARD_SCAN);\n\
    float bsum = gauss1d(dsty - 2.0, HARD_BLOOM_SCAN) + gauss1d(dsty - 1.0, HARD_BLOOM_SCAN) +\n\
                 gauss1d(dsty, HARD_BLOOM_SCAN) + gauss1d(dsty + 1.0, HARD_BLOOM_SCAN) +\n\
                 gauss1d(dsty + 2.0, HARD_BLOOM_SCAN);\n\
    float3 color = lin * wsum + lin * bsum * BLOOM_AMOUNT;\n\
    float mx = fract((floor(pos.x * prm.outSize.x) + 0.5) / 3.0);\n\
    float3 m = float3(MASK_DARK);\n\
    if (mx < 1.0/3.0) m.r = MASK_LIGHT;\n\
    else if (mx < 2.0/3.0) m.g = MASK_LIGHT;\n\
    else m.b = MASK_LIGHT;\n\
    color *= m;\n\
    outTex.write(float4(clamp(delinearize3(color), 0.0, 1.0), 1.0), gid);\n\
}\n";

struct avf_crt {
    struct mp_log *log;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso;
    CVMetalTextureCacheRef tcache;
    CVPixelBufferPoolRef out_pool;
    int out_w, out_h;
};

struct avf_crt *avf_crt_create(struct mp_log *log)
{
    struct avf_crt *crt = talloc_zero(NULL, struct avf_crt);
    crt->log = log;

    crt->dev = MTLCreateSystemDefaultDevice();
    if (!crt->dev)
        goto error;
    crt->queue = [crt->dev newCommandQueue];

    NSError *err = nil;
    id<MTLLibrary> lib = [crt->dev newLibraryWithSource:kernel_src
                                                options:nil error:&err];
    if (!lib) {
        mp_err(log, "CRT shader compile failed: %s\n",
               err.localizedDescription.UTF8String);
        goto error;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"crt_lottes"];
    crt->pso = [crt->dev newComputePipelineStateWithFunction:fn error:&err];
    if (!crt->pso) {
        mp_err(log, "CRT pipeline creation failed: %s\n",
               err.localizedDescription.UTF8String);
        goto error;
    }

    if (CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL, crt->dev, NULL,
                                  &crt->tcache) != kCVReturnSuccess)
        goto error;

    mp_verbose(log, "native CRT (crt-lottes) initialized\n");
    return crt;

error:
    avf_crt_destroy(&crt);
    return NULL;
}

void avf_crt_destroy(struct avf_crt **crtp)
{
    struct avf_crt *crt = *crtp;
    if (!crt)
        return;
    if (crt->out_pool)
        CVPixelBufferPoolRelease(crt->out_pool);
    if (crt->tcache)
        CFRelease(crt->tcache);
    crt->dev = nil;
    crt->queue = nil;
    crt->pso = nil;
    talloc_free(crt);
    *crtp = NULL;
}

static CVMetalTextureRef plane_texture(struct avf_crt *crt, CVPixelBufferRef pb,
                                       int plane, MTLPixelFormat fmt)
{
    CVMetalTextureRef tex = NULL;
    size_t w = CVPixelBufferGetWidthOfPlane(pb, plane);
    size_t h = CVPixelBufferGetHeightOfPlane(pb, plane);
    if (CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
            crt->tcache, pb, NULL, fmt, w, h, plane, &tex) != kCVReturnSuccess)
        return NULL;
    return tex;
}

CVPixelBufferRef avf_crt_process(struct avf_crt *crt, CVPixelBufferRef in,
                                 int out_w, int out_h,
                                 const struct mp_image_params *params)
{
    if (!crt || out_w < 1 || out_h < 1)
        return NULL;

    // input must be biplanar YCbCr (both hw-decode and the sw upload path are)
    OSType fmt = CVPixelBufferGetPixelFormatType(in);
    MTLPixelFormat luma_fmt, chroma_fmt;
    float luma_scale;
    switch (fmt) {
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
        luma_fmt = MTLPixelFormatR8Unorm;
        chroma_fmt = MTLPixelFormatRG8Unorm;
        luma_scale = 1.0f;
        break;
    case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
        luma_fmt = MTLPixelFormatR16Unorm;
        chroma_fmt = MTLPixelFormatRG16Unorm;
        luma_scale = 65535.0f / (64.0f * 1023.0f);
        break;
    default:
        return NULL;
    }

    if (!crt->out_pool || crt->out_w != out_w || crt->out_h != out_h) {
        if (crt->out_pool) {
            CVPixelBufferPoolRelease(crt->out_pool);
            crt->out_pool = NULL;
        }
        NSDictionary *attrs = @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (id)kCVPixelBufferWidthKey: @(out_w),
            (id)kCVPixelBufferHeightKey: @(out_h),
            (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
            (id)kCVPixelBufferMetalCompatibilityKey: @YES,
        };
        if (CVPixelBufferPoolCreate(kCFAllocatorDefault, NULL,
                (__bridge CFDictionaryRef)attrs,
                &crt->out_pool) != kCVReturnSuccess)
            return NULL;
        crt->out_w = out_w;
        crt->out_h = out_h;
    }

    CVPixelBufferRef out = NULL;
    if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, crt->out_pool,
                                           &out) != kCVReturnSuccess || !out)
        return NULL;

    CVMetalTextureRef luma = plane_texture(crt, in, 0, luma_fmt);
    CVMetalTextureRef chroma = plane_texture(crt, in, 1, chroma_fmt);
    CVMetalTextureRef dst = NULL;
    CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, crt->tcache,
        out, NULL, MTLPixelFormatBGRA8Unorm, out_w, out_h, 0, &dst);
    if (!luma || !chroma || !dst)
        goto error;

    struct {
        float videoSize[2];
        float outSize[2];
        float lumaScale;
        int mtx601;
    } prm = {
        .videoSize = {CVPixelBufferGetWidth(in), CVPixelBufferGetHeight(in)},
        .outSize = {out_w, out_h},
        .lumaScale = luma_scale,
        .mtx601 = params && params->repr.sys == PL_COLOR_SYSTEM_BT_601,
    };

    id<MTLCommandBuffer> cb = [crt->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:crt->pso];
    [enc setTexture:CVMetalTextureGetTexture(luma) atIndex:0];
    [enc setTexture:CVMetalTextureGetTexture(chroma) atIndex:1];
    [enc setTexture:CVMetalTextureGetTexture(dst) atIndex:2];
    [enc setBytes:&prm length:sizeof(prm) atIndex:0];
    MTLSize tg = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((out_w + 15) / 16, (out_h + 15) / 16, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    CFRelease(luma);
    CFRelease(chroma);
    CFRelease(dst);
    CVMetalTextureCacheFlush(crt->tcache, 0);
    return out;  // +1

error:
    if (luma)
        CFRelease(luma);
    if (chroma)
        CFRelease(chroma);
    if (dst)
        CFRelease(dst);
    CVPixelBufferRelease(out);
    return NULL;
}
