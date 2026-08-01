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

    @objc init(_ vo: UnsafeMutablePointer<vo>) {
        let log = LogHelper(mp_log_new(vo, vo.pointee.log, "avf"))
        let option = OptionHelper(vo, vo.pointee.global)
        super.init(option, log)
        eventsLock.withLock { self.vo = vo }
        input = InputHelper(vo.pointee.input_ctx, option)

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

        DispatchQueue.main.sync {
            window?.delegate = nil
            window?.close()

            uninitCommon()
        }
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
