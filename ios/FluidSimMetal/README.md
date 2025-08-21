# FluidSimMetal (iOS)

A Metal-based iOS scaffold to port the WebGL Fluid Simulation. Targets 120Hz ProMotion and HDR on iPad.

How to open:
1. On a Mac with Xcode 15+, open this folder as a Swift Package or create a new Xcode project and add this package.
2. Build and run on an iPad (iOS 16+). The app uses MTKView with preferredFramesPerSecond=120 and HDR pixel format.

Next steps to complete the port:
- Translate the GLSL fragment shaders to MSL functions (advection, divergence, curl, vorticity, pressure, gradient subtract, splat, bloom, sunrays).
- Create ping-pong textures for velocity/dye/pressure and per-pass pipelines mirroring the original render order.
- Map multi-touch input to splats.
- Add UI controls for resolution, dissipation, iterations, effects.

Notes on HDR:
- MTKView is configured with colorPixelFormat .bgra10_xr and Display P3 colorspace for extended dynamic range on supported iPads.
- Keep values in linear space; tone-map as needed before display.
