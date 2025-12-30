
.PHONY: build test sim clean

PROJECT = States.xcodeproj
SCHEME = States


# Default SDK (macOS for local dev)
SDK = macosx
DESTINATION = 'platform=macOS'

# iOS Settings
IOS_SDK = iphonesimulator
IOS_DESTINATION = 'platform=iOS Simulator,name=iPhone 17'

build:
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -sdk $(SDK) -destination $(DESTINATION) build

build-ios:
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -sdk $(IOS_SDK) -destination $(IOS_DESTINATION) build

test:
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -sdk $(SDK) -destination $(DESTINATION) test

test-ios:
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -sdk $(IOS_SDK) -destination $(IOS_DESTINATION) test

sim-ios:
	open -a Simulator

clean:
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) clean
