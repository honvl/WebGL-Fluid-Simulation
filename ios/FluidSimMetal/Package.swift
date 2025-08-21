// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FluidSimMetal",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .library(name: "FluidSimMetal", targets: ["FluidSimMetal"])
    ],
    targets: [
        .target(
            name: "FluidSimMetal",
            path: "FluidSimMetal",
            resources: [
                .process("Assets.xcassets")
            ]
        )
    ]
)
