// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "StatesCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "StatesCore",
            targets: ["StatesCore"]
        ),
        // Expose MLX products as libraries if needed (optional, based on instruction wording)
        .library(
            name: "MLX",
            targets: ["MLX"]
        ),
        .library(
            name: "MLXNN",
            targets: ["MLXNN"]
        ),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]
        ),
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/mlx.git", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.22.0")
    ],
    targets: [
        .target(
            name: "StatesCore",
            dependencies: [
                .product(name: "MLX", package: "mlx"),
                .product(name: "MLXNN", package: "mlx"),
                .product(name: "MLXLMCommon", package: "mlx"),
                .product(name: "MLXLLM", package: "mlx"),
                .product(name: "SwiftProtobuf", package: "swift-protobuf"),
                .product(name: "SwiftProtobufPluginLibrary", package: "swift-protobuf"),
            ],
            path: "Sources/StatesCore",
            exclude: [
                // Exclude generated protobuf files if necessary
                "Protos/Generated"
            ],
            sources: [
                // Include all source files except those excluded
                "."
            ]
        )
    ]
)
