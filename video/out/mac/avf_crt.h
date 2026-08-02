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

#pragma once

#include <CoreVideo/CoreVideo.h>
#include <stdbool.h>

struct mp_log;
struct mp_image_params;

// Native Metal port of the crt-lottes user shader for vo_avfoundation:
// converts a decoded NV12/P010 CVPixelBuffer to BGRA at output size with
// the CRT effect applied, so the effect works on the scanout path.
struct avf_crt;

struct avf_crt *avf_crt_create(struct mp_log *log);
void avf_crt_destroy(struct avf_crt **crt);

// Returns a +1 retained BGRA output buffer, or NULL if the input is
// unsupported or processing failed (caller then uses the input as-is).
CVPixelBufferRef avf_crt_process(struct avf_crt *crt, CVPixelBufferRef in,
                                 int out_w, int out_h,
                                 const struct mp_image_params *params);
