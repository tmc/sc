module github.com/tmc/sc/cmd/web-visualizer

go 1.23

toolchain go1.24.4

require (
	github.com/gorilla/mux v1.8.1
	github.com/gorilla/websocket v1.5.1
	github.com/tmc/sc v0.0.0
	google.golang.org/protobuf v1.36.5
)

require (
	github.com/mattn/go-sqlite3 v1.14.32 // indirect
	golang.org/x/exp v0.0.0-20230307190834-24139beb5833 // indirect
	golang.org/x/net v0.35.0 // indirect
	golang.org/x/sys v0.30.0 // indirect
	golang.org/x/text v0.22.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250218202821-56aae31c358a // indirect
	google.golang.org/grpc v1.72.0 // indirect
)

replace github.com/tmc/sc => ../..
