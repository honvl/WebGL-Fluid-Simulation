import Foundation
import Metal
import MetalKit
import simd

public final class Renderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let queue: MTLCommandQueue

    // Buffers
    private var quadVB: MTLBuffer!

    // Pipelines
    private var copyPSO: MTLRenderPipelineState!
    private var colorPSO: MTLRenderPipelineState!
    private var advectionPSO: MTLRenderPipelineState!
    private var splatPSO: MTLRenderPipelineState!
    private var divergencePSO: MTLRenderPipelineState!
    private var curlPSO: MTLRenderPipelineState!
    private var vorticityPSO: MTLRenderPipelineState!
    private var pressurePSO: MTLRenderPipelineState!
    private var gradientSubtractPSO: MTLRenderPipelineState!

    // Samplers
    private var linearSampler: MTLSamplerState!

    // Uniforms
    private var texelSize = simd_float2(1,1)

    // HDR control
    public var enableHDR: Bool = true

    // Simulation textures (ping-pong)
    private struct PingPong { var read: MTLTexture; var write: MTLTexture }
    private var velocity: PingPong?
    private var dye: PingPong?
    private var pressure: PingPong?
    private var divergence: MTLTexture?
    private var curl: MTLTexture?
    private var simResolution: Int = 128
    private var dyeResolution: Int = 1024

    // Touch state
    private struct Pointer { var uv: SIMD2<Float>; var prevUV: SIMD2<Float>; var down: Bool }
    private var pointer = Pointer(uv: .zero, prevUV: .zero, down: false)

    public init(view: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
        self.device = device
        self.queue = device.makeCommandQueue()!
        super.init()

        // Configure MTKView for 120Hz + HDR
        view.device = device
        // Prefer HDR if available; fall back otherwise
        if device.supportsFamily(.apple7) {
            view.colorPixelFormat = .bgra10_xr
            view.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)
        } else {
            view.colorPixelFormat = .bgra8Unorm
            view.colorspace = CGColorSpace(name: CGColorSpace.displayP3)
        }
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 120

        buildResources()
        buildPipelines(view: view)
        allocateTextures(drawableSize: view.drawableSize)

        view.delegate = self
    }

    private func buildResources() {
        let quad: [Float] = [
            -1, -1,
            -1,  1,
             1,  1,
             1, -1
        ]
        quadVB = device.makeBuffer(bytes: quad, length: MemoryLayout<Float>.size * quad.count)

        // Sampler
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        linearSampler = device.makeSamplerState(descriptor: sd)
    }

    private func buildPipelines(view: MTKView) {
        let library = try! device.makeDefaultLibrary(bundle: .module)

        let vfun = library.makeFunction(name: "fullscreenVS")!

        func makePSO(fragment: String, blending: Bool = false) -> MTLRenderPipelineState {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vfun
            desc.fragmentFunction = library.makeFunction(name: fragment)
            desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
            if blending {
                let ca = desc.colorAttachments[0]!
                ca.isBlendingEnabled = true
                ca.rgbBlendOperation = .add
                ca.alphaBlendOperation = .add
                ca.sourceRGBBlendFactor = .one
                ca.sourceAlphaBlendFactor = .one
                ca.destinationRGBBlendFactor = .one
                ca.destinationAlphaBlendFactor = .one
            }
            return try! device.makeRenderPipelineState(descriptor: desc)
        }

    copyPSO = makePSO(fragment: "copyFS")
    colorPSO = makePSO(fragment: "colorFS")
    advectionPSO = makePSO(fragment: "advectionFS")
    splatPSO = makePSO(fragment: "splatFS")
    divergencePSO = makePSO(fragment: "divergenceFS")
    curlPSO = makePSO(fragment: "curlFS")
    vorticityPSO = makePSO(fragment: "vorticityFS")
    pressurePSO = makePSO(fragment: "pressureFS")
    gradientSubtractPSO = makePSO(fragment: "gradientSubtractFS")
    }

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        texelSize = simd_float2(1.0 / Float(size.width), 1.0 / Float(size.height))
    allocateTextures(drawableSize: size)
    }

    public func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { return }

        // Update simulation step: simple advection of dye by velocity, and apply current pointer splat
        step(view: view, commandBuffer: cb)

        // Display dye to screen
        enc.setRenderPipelineState(copyPSO)
        var u = texelSize
        enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        enc.setVertexBuffer(quadVB, offset: 0, index: 0)
        if let dye = dye?.read {
            enc.setFragmentTexture(dye, index: 0)
            enc.setFragmentSamplerState(linearSampler, index: 0)
        }
        enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)

        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }

    private func allocateTextures(drawableSize: CGSize) {
        // Compute sim/dye target sizes maintaining aspect like the web version
        let aspect = max(drawableSize.width / max(1, drawableSize.height), 1)
        let simW = Int(max(32, round(CGFloat(simResolution) * aspect)))
        let simH = Int(max(32, round(CGFloat(simResolution))))
        let dyeW = Int(max(64, round(CGFloat(dyeResolution) * aspect)))
        let dyeH = Int(max(64, round(CGFloat(dyeResolution))))

        func makeTex(w: Int, h: Int, format: MTLPixelFormat) -> MTLTexture {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            desc.usage = [.shaderRead, .renderTarget]
            desc.storageMode = .private
            return device.makeTexture(descriptor: desc)!
        }

        func makePair(w: Int, h: Int, format: MTLPixelFormat) -> PingPong {
            return PingPong(read: makeTex(w: w, h: h, format: format),
                            write: makeTex(w: w, h: h, format: format))
        }

    // Velocity = RG16Float, Dye = RGBA16Float, Pressure/R/Curl = R16Float
    velocity = makePair(w: simW, h: simH, format: .rg16Float)
    dye = makePair(w: dyeW, h: dyeH, format: .rgba16Float)
    pressure = makePair(w: simW, h: simH, format: .r16Float)
    divergence = makeTex(w: simW, h: simH, format: .r16Float)
    curl = makeTex(w: simW, h: simH, format: .r16Float)
    }

    private func swap(_ pp: inout PingPong) { let tmp = pp.read; pp.read = pp.write; pp.write = tmp }

    private func renderTo(texture: MTLTexture, _ body: (MTLRenderCommandEncoder) -> Void) {
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = texture
        rpd.colorAttachments[0].loadAction = .dontCare
        rpd.colorAttachments[0].storeAction = .store
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { return }
        body(enc)
        enc.endEncoding()
        cb.commit()
    }

    private func step(view: MTKView, commandBuffer: MTLCommandBuffer) {
    guard var velocity = velocity, var dye = dye, var pressure = pressure, let divergence = divergence, let curl = curl else { return }

        // Apply a splat when touching
        if pointer.down {
            let aspect = Float(dye.read.width) / Float(max(1, dye.read.height))
            // Splat into velocity
            renderTo(texture: velocity.write) { enc in
                enc.setRenderPipelineState(splatPSO)
                var u = simd_float2(1.0 / Float(velocity.read.width), 1.0 / Float(velocity.read.height))
                enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
                enc.setVertexBuffer(quadVB, offset: 0, index: 0)
                struct SplatU { var aspectRatio: Float; var point: SIMD2<Float>; var radius: Float; var color: SIMD3<Float> }
                var su = SplatU(aspectRatio: aspect, point: pointer.uv, radius: 0.01, color: SIMD3<Float>((pointer.uv - pointer.prevUV) * 6000, 0))
                enc.setFragmentBytes(&su, length: MemoryLayout<SplatU>.size, index: 0)
                enc.setFragmentTexture(velocity.read, index: 0)
                enc.setFragmentSamplerState(linearSampler, index: 0)
                enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
            }
            swap(&self.velocity!)

            // Splat into dye
            renderTo(texture: dye.write) { enc in
                enc.setRenderPipelineState(splatPSO)
                var u = simd_float2(1.0 / Float(dye.read.width), 1.0 / Float(dye.read.height))
                enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
                enc.setVertexBuffer(quadVB, offset: 0, index: 0)
                struct SplatU { var aspectRatio: Float; var point: SIMD2<Float>; var radius: Float; var color: SIMD3<Float> }
                var color = SIMD3<Float>(0.8, 0.2, 1.2)
                var su = SplatU(aspectRatio: aspect, point: pointer.uv, radius: 0.01, color: color)
                enc.setFragmentBytes(&su, length: MemoryLayout<SplatU>.size, index: 0)
                enc.setFragmentTexture(dye.read, index: 0)
                enc.setFragmentSamplerState(linearSampler, index: 0)
                enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
            }
            swap(&self.dye!)
        }

        // Compute curl
        renderTo(texture: curl) { enc in
            enc.setRenderPipelineState(curlPSO)
            var u = simd_float2(1.0 / Float(velocity.read.width), 1.0 / Float(velocity.read.height))
            enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
            enc.setVertexBuffer(quadVB, offset: 0, index: 0)
            enc.setFragmentTexture(velocity.read, index: 0)
            enc.setFragmentSamplerState(linearSampler, index: 0)
            enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
        }

        // Vorticity confinement
        renderTo(texture: velocity.write) { enc in
            enc.setRenderPipelineState(vorticityPSO)
            var u = simd_float2(1.0 / Float(velocity.read.width), 1.0 / Float(velocity.read.height))
            enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
            enc.setVertexBuffer(quadVB, offset: 0, index: 0)
            struct VU { var curl: Float; var dt: Float }
            var vu = VU(curl: 30, dt: 1.0/120.0)
            enc.setFragmentBytes(&vu, length: MemoryLayout<VU>.size, index: 0)
            enc.setFragmentTexture(velocity.read, index: 0)
            enc.setFragmentTexture(curl, index: 1)
            enc.setFragmentSamplerState(linearSampler, index: 0)
            enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
        }
        swap(&self.velocity!)

        // Divergence
        renderTo(texture: divergence) { enc in
            enc.setRenderPipelineState(divergencePSO)
            var u = simd_float2(1.0 / Float(velocity.read.width), 1.0 / Float(velocity.read.height))
            enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
            enc.setVertexBuffer(quadVB, offset: 0, index: 0)
            enc.setFragmentTexture(velocity.read, index: 0)
            enc.setFragmentSamplerState(linearSampler, index: 0)
            enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
        }

        // Clear pressure toward config pressure value (use multiply by 0.8 similar to JS default)
        renderTo(texture: pressure.write) { enc in
            enc.setRenderPipelineState(copyPSO) // initialize from read then multiply is skipped for simplicity
            var u = simd_float2(1.0 / Float(pressure.read.width), 1.0 / Float(pressure.read.height))
            enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
            enc.setVertexBuffer(quadVB, offset: 0, index: 0)
            enc.setFragmentTexture(pressure.read, index: 0)
            enc.setFragmentSamplerState(linearSampler, index: 0)
            enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
        }
        swap(&self.pressure!)

        // Pressure iterations
        let iterations = 20
        for _ in 0..<iterations {
            renderTo(texture: pressure.write) { enc in
                enc.setRenderPipelineState(pressurePSO)
                var u = simd_float2(1.0 / Float(pressure.read.width), 1.0 / Float(pressure.read.height))
                enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
                enc.setVertexBuffer(quadVB, offset: 0, index: 0)
                enc.setFragmentTexture(pressure.read, index: 0)
                enc.setFragmentTexture(divergence, index: 1)
                enc.setFragmentSamplerState(linearSampler, index: 0)
                enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
            }
            swap(&self.pressure!)
        }

        // Subtract gradient
        renderTo(texture: velocity.write) { enc in
            enc.setRenderPipelineState(gradientSubtractPSO)
            var u = simd_float2(1.0 / Float(velocity.read.width), 1.0 / Float(velocity.read.height))
            enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
            enc.setVertexBuffer(quadVB, offset: 0, index: 0)
            enc.setFragmentTexture(pressure.read, index: 0)
            enc.setFragmentTexture(velocity.read, index: 1)
            enc.setFragmentSamplerState(linearSampler, index: 0)
            enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
        }
        swap(&self.velocity!)

        // Advect velocity
        renderTo(texture: velocity.write) { enc in
            enc.setRenderPipelineState(advectionPSO)
            var u = simd_float2(1.0 / Float(velocity.read.width), 1.0 / Float(velocity.read.height))
            enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
            enc.setVertexBuffer(quadVB, offset: 0, index: 0)
            struct AdvU { var texelSize: SIMD2<Float>; var dt: Float; var dissipation: Float }
            var au = AdvU(texelSize: u, dt: 1.0/120.0, dissipation: 0.2)
            enc.setFragmentBytes(&au, length: MemoryLayout<AdvU>.size, index: 0)
            enc.setFragmentTexture(velocity.read, index: 0)
            enc.setFragmentTexture(velocity.read, index: 1)
            enc.setFragmentSamplerState(linearSampler, index: 0)
            enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
        }
        swap(&self.velocity!)

        // Advect dye by velocity
        renderTo(texture: dye.write) { enc in
            enc.setRenderPipelineState(advectionPSO)
            var u = simd_float2(1.0 / Float(dye.read.width), 1.0 / Float(dye.read.height))
            enc.setVertexBytes(&u, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
            enc.setVertexBuffer(quadVB, offset: 0, index: 0)
            struct AdvU { var texelSize: SIMD2<Float>; var dt: Float; var dissipation: Float }
            var au = AdvU(texelSize: u, dt: 1.0/120.0, dissipation: 1.0)
            enc.setFragmentBytes(&au, length: MemoryLayout<AdvU>.size, index: 0)
            enc.setFragmentTexture(self.velocity!.read, index: 0)
            enc.setFragmentTexture(dye.read, index: 1)
            enc.setFragmentSamplerState(linearSampler, index: 0)
            enc.drawPrimitives(type: .triangleFan, vertexStart: 0, vertexCount: 4)
        }
        swap(&self.dye!)
    }

    // Exposed touch handlers for the SwiftUI wrapper
    public func pointerDown(at uv: SIMD2<Float>) { pointer.down = true; pointer.prevUV = uv; pointer.uv = uv }
    public func pointerMove(to uv: SIMD2<Float>) { pointer.prevUV = pointer.uv; pointer.uv = uv }
    public func pointerUp() { pointer.down = false }
}
