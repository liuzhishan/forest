fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iface_files = &[
        "protos/meta.proto",
        "protos/feature.proto",
        "protos/service.proto",
    ];
    let dirs = &["."];

    println!("start build proto");

    tonic_build::configure()
        .build_client(true)
        .compile(iface_files, dirs)
        .unwrap_or_else(|e| panic!("protobuf compilation failed: {}", e));

    for file in iface_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    Ok(())
}