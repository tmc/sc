fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_files = [
        "../../proto/statecharts/v1/statecharts.proto",
        "../../proto/statecharts/v1/statechart_service.proto",
        "../../proto/validation/v1/validator.proto",
    ];

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .format(true)
        .out_dir("src/generated")
        .compile(&proto_files, &["../../proto"])?;

    // Tell cargo to rerun this build script if the proto files change
    for proto_file in proto_files.iter() {
        println!("cargo:rerun-if-changed={}", proto_file);
    }

    Ok(())
}