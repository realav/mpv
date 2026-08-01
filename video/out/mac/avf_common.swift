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
    var osdLayer: CALayer?

    @objc init(_ vo: UnsafeMutablePointer<vo>) {
        let log = LogHelper(mp_log_new(vo, vo.pointee.log, "avf"))
        let option = OptionHelper(vo, vo.pointee.global)
        super.init(option, log)
        eventsLock.withLock { self.vo = vo }
        input = InputHelper(vo.pointee.input_ctx, option)

        DispatchQueue.main.sync {
            let layer = AVSampleBufferDisplayLayer()
            // mpv positions/letterboxes via the window; the layer only scales
            layer.videoGravity = .resizeAspect
            layer.backgroundColor = NSColor.black.cgColor
            self.layer = layer

            // transparent overlay for mpv's OSD/subtitle bitmaps
            let osd = CALayer()
            osd.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            osd.actions = ["contents": NSNull(), "bounds": NSNull(), "position": NSNull()]
            layer.addSublayer(osd)
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
            guard let layer = self.layer else {
                log.error("Something went wrong, no AVSampleBufferDisplayLayer was initialized")
                exit(1)
            }

            if window == nil {
                initView(vo, layer)
                initWindow(vo, previousActiveApp)
                initWindowState()
            }

            if forcePosition {
                window?.updateFrame(wr, screen)
            } else if option.vo.auto_window_resize {
                window?.updateSize(wr.size)
            }

            if option.vo.focus_on == 2 {
                NSApp.activate(ignoringOtherApps: true)
            }

            updateOsdGeometry()
            windowDidResize()
        }

        return true
    }

    private func updateOsdGeometry() {
        guard let layer = self.layer, let osd = self.osdLayer else { return }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        osd.frame = layer.bounds
        osd.contentsScale = window?.backingScaleFactor ?? 1
        CATransaction.commit()
    }

    // takes ownership of a +1 retained CGImageRef (nil clears the OSD)
    @objc func setOsd(_ image: UnsafeMutableRawPointer?) {
        let img = image.map { Unmanaged<CGImage>.fromOpaque($0).takeRetainedValue() }
        DispatchQueue.main.async {
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            self.osdLayer?.contents = img
            CATransaction.commit()
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
        layer?.contentsScale = window?.backingScaleFactor ?? 1
        updateOsdGeometry()
        windowDidResize()
    }

    override func windowDidChangeOcclusionState() {
        flagEvents(VO_EVENT_EXPOSE)
    }
}
