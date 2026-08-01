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

import Cocoa
import AVFoundation

/// Window host for vo_avfoundation: mpv's usual Cocoa window/input handling
/// with an AVSampleBufferDisplayLayer as the content layer. Video frames are
/// enqueued directly; macOS performs all HDR/EDR tone mapping.
class AVFCommon: Common {
    @objc var layer: AVSampleBufferDisplayLayer?
    var rootLayer: CALayer?
    var osdLayer: CALayer?
    var osdPixelBuffer: CVPixelBuffer?
    var edrWarmup: CAMetalLayer?

    // display-link vsync pacing, same as MacCommon: flip blocks until the
    // next display refresh so mpv's display-sync sees real vsync cadence
    var presentation: Presentation?
    var timer: PreciseTimer?
    var swapTime: UInt64 = 0
    let swapLock: NSCondition = NSCondition()

    @objc init(_ vo: UnsafeMutablePointer<vo>) {
        let log = LogHelper(mp_log_new(vo, vo.pointee.log, "avf"))
        let option = OptionHelper(vo, vo.pointee.global)
        super.init(option, log)
        eventsLock.withLock { self.vo = vo }
        input = InputHelper(vo.pointee.input_ctx, option)
        presentation = Presentation(common: self)
        timer = PreciseTimer(common: self)

        DispatchQueue.main.sync {
            // plain container as the view's backing layer; the video and OSD
            // layers are siblings, so OSD updates composite on the GPU instead
            // of being CPU-converted into the video layer's EDR colorspace
            let root = CALayer()
            root.backgroundColor = NSColor.black.cgColor
            self.rootLayer = root

            let layer = AVSampleBufferDisplayLayer()
            // mpv positions/letterboxes via the window; the layer only scales
            layer.videoGravity = .resizeAspect
            layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            root.addSublayer(layer)
            self.layer = layer

            // transparent overlay for mpv's OSD/subtitle bitmaps; sized and
            // positioned per update to the OSD's bounding box
            let osd = CALayer()
            osd.actions = ["contents": NSNull(), "bounds": NSNull(), "position": NSNull()]
            root.addSublayer(osd)
            self.osdLayer = osd

            initMisc(vo)
        }
    }

    @objc func config(_ vo: UnsafeMutablePointer<vo>) -> Bool {
        eventsLock.withLock { self.vo = vo }

        DispatchQueue.main.sync {
            let previousActiveApp = getActiveApp()
            initApp()

            let (screen, wr, forcePosition) = getInitProperties(vo)
            guard let root = self.rootLayer, let layer = self.layer else {
                log.error("Something went wrong, no AVSampleBufferDisplayLayer was initialized")
                exit(1)
            }

            if window == nil {
                initView(vo, root)
                initWindow(vo, previousActiveApp)
                initWindowState()
                layer.frame = root.bounds
            }

            if forcePosition {
                window?.updateFrame(wr, screen)
            } else if option.vo.auto_window_resize {
                window?.updateSize(wr.size)
            }

            if option.vo.focus_on == 2 {
                NSApp.activate(ignoringOtherApps: true)
            }

            windowDidResize()
        }

        return true
    }

    // takes ownership of a +1 retained CVPixelBufferRef (nil clears the OSD);
    // the buffer is IOSurface-backed, so CoreAnimation displays it zero-copy
    @objc func setOsd(_ pixelBuffer: UnsafeMutableRawPointer?) {
        let pb = pixelBuffer.map { Unmanaged<CVPixelBuffer>.fromOpaque($0).takeRetainedValue() }
        DispatchQueue.main.async {
            guard let osd = self.osdLayer, let root = self.rootLayer else { return }
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            if let pb, let surface = CVPixelBufferGetIOSurface(pb)?.takeUnretainedValue() {
                osd.contentsScale = self.window?.backingScaleFactor ?? 1
                osd.frame = root.bounds
                osd.contents = surface
            } else {
                osd.contents = nil
            }
            CATransaction.commit()
            self.osdPixelBuffer = pb  // keep the surface alive while displayed
        }
    }

    @objc func uninit(_ vo: UnsafeMutablePointer<vo>) {
        window?.waitForAnimation()

        timer?.terminate()

        DispatchQueue.main.sync {
            window?.delegate = nil
            window?.close()

            uninitCommon()
        }
    }

    @objc func swapBuffer() {
        if option.mac.macos_render_timer > RENDER_TIMER_SYSTEM {
            swapLock.lock()
            while swapTime < 1 {
                swapLock.wait()
            }
            swapTime = 0
            swapLock.unlock()
        }
    }

    @objc func fillVsync(info: UnsafeMutablePointer<vo_vsync_info>) {
        if option.mac.macos_render_timer != RENDER_TIMER_PRESENTATION_FEEDBACK { return }

        let next = presentation?.next()
        info.pointee.vsync_duration = next?.duration ?? -1
        info.pointee.skipped_vsyncs = next?.skipped ?? -1
        info.pointee.last_queue_display_time = next?.time ?? -1
    }

    override func displayLinkCallback(_ displayLink: CVDisplayLink,
                                      _ inNow: UnsafePointer<CVTimeStamp>,
                                      _ inOutputTime: UnsafePointer<CVTimeStamp>,
                                      _ flagsIn: CVOptionFlags,
                                      _ flagsOut: UnsafeMutablePointer<CVOptionFlags>) -> CVReturn {
        let signalSwap = {
            self.swapLock.lock()
            self.swapTime += 1
            self.swapLock.signal()
            self.swapLock.unlock()
        }

        if option.mac.macos_render_timer > RENDER_TIMER_SYSTEM {
            if let timer = self.timer, option.mac.macos_render_timer == RENDER_TIMER_PRECISE {
                timer.scheduleAt(time: inOutputTime.pointee.hostTime, closure: signalSwap)
                return kCVReturnSuccess
            }

            signalSwap()
            return kCVReturnSuccess
        }

        if option.mac.macos_render_timer == RENDER_TIMER_PRESENTATION_FEEDBACK {
            presentation?.add(time: inOutputTime.pointee)
        }

        return kCVReturnSuccess
    }

    override func startDisplayLink(_ vo: UnsafeMutablePointer<vo>) {
        super.startDisplayLink(vo)
        timer?.updatePolicy(periodSeconds: 1 / currentFps())
    }

    override func updateDisplaylink() {
        super.updateDisplaylink()
        timer?.updatePolicy(periodSeconds: 1 / currentFps())
    }

    // opaque pointer to keep CoreMedia types out of the generated ObjC header
    @objc func enqueue(_ sampleBuffer: UnsafeMutableRawPointer) {
        let sampleBuffer = Unmanaged<CMSampleBuffer>.fromOpaque(sampleBuffer).takeUnretainedValue()
        guard let layer = self.layer else { return }
        if layer.status == .failed {
            log.warning("AVSampleBufferDisplayLayer failed, flushing: " +
                        (layer.error?.localizedDescription ?? "unknown error"))
            layer.flush()
        }
        layer.enqueue(sampleBuffer)
    }

    @objc func flush() {
        layer?.flushAndRemoveImage()
    }

    // ask macOS to switch the display into EDR mode while the window is still
    // black, so the visible brightness remap doesn't flash over the video
    @objc func setWantsEdr(_ wants: Bool) {
        DispatchQueue.main.async {
            if wants, self.edrWarmup == nil, let root = self.rootLayer {
                let warmup = CAMetalLayer()
                warmup.frame = CGRect(x: 0, y: 0, width: 1, height: 1)
                warmup.wantsExtendedDynamicRangeContent = true
                warmup.isOpaque = false
                root.insertSublayer(warmup, at: 0)
                self.edrWarmup = warmup
            } else if !wants, let warmup = self.edrWarmup {
                warmup.removeFromSuperlayer()
                self.edrWarmup = nil
            }
        }
    }

    override func windowDidResize() {
        flagEvents(VO_EVENT_RESIZE | VO_EVENT_EXPOSE)
    }

    override func windowDidChangeBackingProperties() {
        let scale = window?.backingScaleFactor ?? 1
        rootLayer?.contentsScale = scale
        layer?.contentsScale = scale
        windowDidResize()
    }

    override func windowDidChangeOcclusionState() {
        flagEvents(VO_EVENT_EXPOSE)
    }
}
