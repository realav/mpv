# mpv fork: `vo-avfoundation`

Custom mpv fork (macOS) adding a **native AVFoundation video output** for
QuickTime-quality HDR at a fraction of the CPU. Daily driver on this machine.

- Fork: `github.com/realav/mpv`, branch `vo-avfoundation`, clone at `~/mpv`
- Upstream remote: `mpv-player/mpv` (`upstream`)
- Installed app: `/Applications/mpv.app` (pre-fork backup: `/Applications/mpv-stock.app`;
  `/usr/local/bin/mpv` symlinks into the app)

## Why this exists

Playing HDR/DV HEVC 4K in stock mpv cost ~100%+ CPU (`vo=libmpv` runs deprecated
OpenGL through Apple's `AppleMetalOpenGLRenderer` shim) and `vo=gpu-next`+Vulkan
(MoltenVK) renders HDR dim (libplacebo tone-maps in-shader; libplacebo has no
Metal backend; libmpv's public render API is OpenGL-only, which is also why IINA
is stuck there). The fix — same approach as KSPlayer/QuickTime — is to hand
decoded frames to macOS and let the OS do HDR tone mapping natively.

## What `vo=avfoundation` does

`video/out/vo_avfoundation.m` + `video/out/mac/avf_common.swift` (plus a 3-line
hook in `video/out/vo.c` and 4 lines in `meson.build` — deliberately minimal for
clean rebases):

- **Video**: VideoToolbox-decoded `CVPixelBuffer` → `CMSampleBuffer` →
  `AVSampleBufferDisplayLayer` (display-immediately). macOS does HDR/EDR tone
  mapping. Software-decoded frames (codecs VT can't do) are uploaded into
  IOSurface-backed NV12/P010 buffers and displayed the same way — the VO plays
  everything; hw decode is simply preferred. Colorimetry (primaries, transfer,
  YCbCr matrix) is tagged on buffers when missing.
- **OSD/subtitles**: per-part GPU-composited `CALayer`s. Each OSD item's packed
  BGRA atlas is memcpy'd to a per-item double-buffered IOSurface (skipped when
  the item's `change_id` is unchanged); parts crop via `contentsRect`, CA does
  all scaling/blending. Updates capped at 60 Hz (`mp_time_ns` gate) — the cap
  bounds core-side ASS rasterization of scripted OSCs, not the layer path.
- **Timing**: flips block on a CVDisplayLink (ported from `MacCommon`), plus
  `get_vsync`. Without this, `video-sync=display-resample` ran video at 2.5x.
  Even with it, display-resample misjudges vsync on this VO (no GPU presentation
  feedback) → use `video-sync=audio` with it (config handles this).
- **EDR pre-warm**: a 1×1 `CAMetalLayer` with `wantsExtendedDynamicRangeContent`
  is added on HDR reconfig so the display's EDR transition happens under the
  black window, not over the first frames.
- Window/input reuse mpv's Swift `Common` infra (`AVFCommon` subclass); the OSD
  layer tree is: plain root container → video layer + OSD part layers as
  siblings (OSD inside the EDR video layer forces CPU sRGB→PQ conversion —
  never nest it there).

Also fixed on this branch (upstream bugs, PR-worthy):

- `osdep/mac/input_helper.swift`: files opened via Finder at launch arrived
  before the core's input_ctx and were dropped → buffered and replayed.
- `input/dnd.c`: the drop handler observed `dropped-files` before its enable
  flag → launch-time drops were discarded → re-read the property when the flag
  turns on. (Together these fixed "double-click opens an empty player".)

Perf (M-series, 4K HDR10 60fps): ~7–15% CPU playback, 0 dropped frames; worst
case ~45% with modernx OSC pinned visible (cost is core-side ASS rasterization,
common to all VOs). 1080p SDR: ~9% vs ~14% on gpu-next.

## Config (`~/.config/mpv/mpv.conf`)

Original pre-project backup: `~/.config/mpv/mpv.conf.bak-20260715232137`.

- Global: `vo=avfoundation` for ALL content + `video-sync=audio`, `hwdec=auto`,
  `force-window=yes` (NOT `immediate` — avoids a gpu-next splash window swap).
- `[shader-active]`: `profile-cond=next(p["glsl-shaders"] or {}) ~= nil` →
  `vo=gpu-next` + `video-sync=display-resample` while any GLSL shader is active
  (the `Meta+c` CRT toggle in `input.conf` triggers it); restores on toggle-off.
  GLSL shaders can never run on the scanout VO (no render pipeline — that's the
  point).
- `[hdr]`: sticky bt.2020 condition (a `nil` primaries reading during decoder
  init must NOT deactivate the profile, else the VO bounces → rapid startup
  flicker). Now only carries cosmetic overrides.
- **Config gotcha**: `#` starts a comment even mid-line in mpv.conf — never use
  Lua's `#` operator in `profile-cond`.

## Updating to a new mpv release

```sh
cd ~/mpv
git fetch upstream && git rebase upstream/master
ninja -C build                      # meson build dir already configured
TOOLS/osxbundle.py build/mpv        # produces build/mpv.app (bundles MoltenVK)
rm -rf /Applications/mpv.app && cp -R build/mpv.app /Applications/mpv.app
```

Build deps (brew): meson, ninja, luajit + existing ffmpeg/libass/libplacebo/
molten-vk. First-time configure was `meson setup build -Dvulkan=enabled -Dlua=luajit`.

## Known limits / notes

- Dolby Vision plays via its HDR10-compatible base layer (verified good for
  profile 8.1). Apple's licensed DV engine is reachable only through a real
  `AVPlayer` (the AetherEngine local-HLS trick) — out of scope.
- `~/code/Archive/mpv-build/` is the pre-fork build script + `noshadow.patch`
  (fullscreen window-shadow artifact fix for macOS Tahoe). Applied to the fork
  once, then reverted on request; re-apply with
  `git apply ~/code/Archive/mpv-build/noshadow.patch` if wanted.
- MoltenVK env var for non-bundle builds:
  `VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`
  (set session-wide via `launchctl setenv`; the .app bundles its own).
- Debugging: `log-file=~/.config/mpv/mpv_logs.txt` (overwritten per run);
  profile CPU with `sample <pid> 3`; check EDR state via
  `NSScreen.maximumExtendedDynamicRangeColorComponentValue`.
