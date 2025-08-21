import SwiftUI
import MetalKit

public struct MetalView: UIViewRepresentable {
    public init() {}
    public func makeUIView(context: Context) -> MTKView {
        let view = TouchMTKView()
        let renderer = Renderer(view: view)
        context.coordinator.renderer = renderer
        view.touchDelegate = renderer
        return view
    }
    public func updateUIView(_ uiView: MTKView, context: Context) {}
    public func makeCoordinator() -> Coordinator { Coordinator() }
    public final class Coordinator { var renderer: Renderer? }
}

final class TouchMTKView: MTKView {
    weak var touchDelegate: Renderer?
    private func uv(from point: CGPoint) -> SIMD2<Float> {
        let u = Float(point.x / max(1, bounds.width))
        let v = Float(1.0 - point.y / max(1, bounds.height))
        return SIMD2<Float>(u, v)
    }
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first else { return }
        touchDelegate?.pointerDown(at: uv(from: t.location(in: self)))
    }
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first else { return }
        touchDelegate?.pointerMove(to: uv(from: t.location(in: self)))
    }
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) { touchDelegate?.pointerUp() }
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) { touchDelegate?.pointerUp() }
}

@main
struct FluidSimMetalApp: App {
    var body: some Scene {
        WindowGroup {
            MetalView().ignoresSafeArea()
        }
    }
}
