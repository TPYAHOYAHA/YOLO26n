import SwiftUI
import AVFoundation
import UniformTypeIdentifiers
import Darwin

@main
struct CoinDetectorApp: App {
    var body: some Scene { WindowGroup { ScannerView() } }
}

final class Scanner: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var coins: [Coin] = []
    @Published var message = ""
    @Published var supported: [Int] = []
    @Published var loading = false
    @Published var stats = "—"
    @Published var aspect = CGFloat(9) / 16
    @Published var backend = ""
    let camera = AVCaptureSession()
    private let queue = DispatchQueue(label: "coin.capture", qos: .userInitiated)
    private var configured = false
    private var detector: Detector?
    private var tracker = Stabilizer()
    private var size = 416
    private var fps = 10
    private var confidence: Float = 0.25
    private var iou = 0.45
    private var stabilized = true
    private var lastStart = -Double.infinity
    private var sampleTime = 0.0
    private var previousCPU = 0.0
    private var count = 0
    private var inferenceFailed = false

    private func report(_ text: String) {
        DispatchQueue.main.async { self.message = text }
    }

    func cameraActive(_ active: Bool) {
        if !active {
            queue.async {
                if self.camera.isRunning { self.camera.stopRunning() }
                self.tracker = Stabilizer()
                self.count = 0
                self.sampleTime = 0
                DispatchQueue.main.async { self.coins = []; self.stats = "—" }
            }
            return
        }
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { self.report("設定でカメラへのアクセスを許可してください。"); return }
            self.queue.async {
                do {
                    if !self.configured { try self.configure() }
                    if !self.camera.isRunning { self.camera.startRunning() }
                } catch { self.report(error.localizedDescription) }
            }
        }
    }

    private func configure() throws {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        else { throw CoinError("背面カメラが見つかりません。") }
        let input = try AVCaptureDeviceInput(device: device)
        let output = AVCaptureVideoDataOutput()
        camera.beginConfiguration()
        defer { camera.commitConfiguration() }
        if camera.canSetSessionPreset(.hd1280x720) { camera.sessionPreset = .hd1280x720 }
        guard camera.canAddInput(input), camera.canAddOutput(output)
        else { throw CoinError("カメラを構成できません。") }
        camera.addInput(input)
        camera.addOutput(output)
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        if let connection = output.connection(with: .video), connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        guard device.activeFormat.videoSupportedFrameRateRanges.contains(where: {
            $0.minFrameRate <= 30 && $0.maxFrameRate >= 30
        }) else {
            camera.removeOutput(output)
            camera.removeInput(input)
            throw CoinError("現在のカメラ形式では30fps固定にできません。")
        }
        device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        output.setSampleBufferDelegate(self, queue: queue)
        configured = true
    }

    func configureInference(size: Int, fps: Int, confidence: Double, iou: Double, stabilized: Bool) {
        queue.async {
            let next = self.detector?.supported.contains(size) == false ? self.detector!.supported[0] : size
            if next != self.size || Float(confidence) != self.confidence ||
                iou != self.iou || stabilized != self.stabilized {
                self.tracker = Stabilizer()
                DispatchQueue.main.async { self.coins = [] }
            }
            self.size = next
            self.fps = fps
            self.confidence = Float(confidence)
            self.iou = iou
            self.stabilized = stabilized
        }
    }

    func load(_ source: URL? = nil, preferred: Int) {
        guard !loading else { return }
        loading = true
        queue.async {
            defer { DispatchQueue.main.async { self.loading = false } }
            do {
                let manager = FileManager.default
                let directory = try manager.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                appropriateFor: nil, create: true)
                let current = directory.appendingPathComponent("model.onnx")
                let staged = directory.appendingPathComponent(UUID().uuidString + ".onnx")
                defer { try? manager.removeItem(at: staged) }
                if let source {
                    let accessed = source.startAccessingSecurityScopedResource()
                    defer { if accessed { source.stopAccessingSecurityScopedResource() } }
                    try manager.copyItem(at: source, to: staged)
                    // Validate the new file before replacing the previous working model.
                    let candidate = try Detector(url: staged, preferred: preferred)
                    if manager.fileExists(atPath: current.path) {
                        _ = try manager.replaceItemAt(current, withItemAt: staged)
                    } else {
                        try manager.moveItem(at: staged, to: current)
                    }
                    candidate.relocate(to: current)
                    self.detector = candidate
                } else {
                    guard manager.fileExists(atPath: current.path) else { return }
                    self.detector = try Detector(url: current, preferred: preferred)
                }
                guard let model = self.detector else { return }
                self.size = model.supported.contains(preferred) ? preferred : model.supported[0]
                self.tracker = Stabilizer()
                self.inferenceFailed = false
                self.count = 0
                self.sampleTime = 0
                DispatchQueue.main.async {
                    self.supported = model.supported
                    self.message = ""
                    self.coins = []
                }
            } catch { self.report("モデルを読み込めません。\n\(error.localizedDescription)") }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let detector, !inferenceFailed, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastStart >= 1 / Double(fps) - 0.004 else { return }
        lastStart = now
        do {
            let found = try detector.detect(buffer, size: size, confidence: confidence)
            let completed = ProcessInfo.processInfo.systemUptime
            let coins = stabilized ? tracker.update(found, time: completed, iou: iou) : found
            let ratio = CGFloat(CVPixelBufferGetWidth(buffer)) / CGFloat(CVPixelBufferGetHeight(buffer))
            let currentBackend = detector.coreML ? "Core ML EP" : "CPU"
            count += 1
            let elapsed = completed - sampleTime
            var metrics: String?
            if elapsed >= 1 {
                let cpuTime = processCPUTime()
                let cpu = sampleTime > 0 ? String(format: "%.0f%%", (cpuTime - previousCPU) / elapsed * 100) : "—"
                let rate = sampleTime > 0 ? String(format: "%.1f", Double(count) / elapsed) : "—"
                metrics = "\(size)×\(size)   \(Int((completed - now) * 1000)) ms   \(rate) FPS\nCPU \(cpu)   メモリ \(processMemory())"
                count = 0
                sampleTime = completed
                previousCPU = cpuTime
            }
            DispatchQueue.main.async {
                self.coins = coins
                self.aspect = ratio
                self.backend = currentBackend
                if let metrics { self.stats = metrics }
            }
        } catch {
            inferenceFailed = true
            DispatchQueue.main.async { self.coins = []; self.stats = "—"; self.backend = "" }
            report("推論を続けられません。モデルを選び直してください。\n\(error.localizedDescription)")
        }
    }
}

private func processCPUTime() -> Double {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
    return Double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec)
        + Double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1_000_000
}

private func processMemory() -> String {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? "\(info.phys_footprint / 1_048_576) MiB" : "—"
}

private final class PreviewSurface: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    override func layoutSubviews() {
        super.layoutSubviews()
        if let connection = preview.connection, connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
    }
}

private struct Preview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewSurface {
        let view = PreviewSurface()
        view.preview.session = session
        view.preview.videoGravity = .resizeAspect
        return view
    }
    func updateUIView(_ view: PreviewSurface, context: Context) { view.setNeedsLayout() }
}

struct ScannerView: View {
    @StateObject private var scanner = Scanner()
    @Environment(\.scenePhase) private var phase
    @AppStorage("size") private var size = 416
    @AppStorage("fps") private var fps = 10
    @AppStorage("confidence") private var confidence = 0.25
    @AppStorage("trackingIoU") private var iou = 0.45
    @AppStorage("stabilized") private var stabilized = true
    @State private var settings = false
    @State private var importing = false
    @State private var reset = false

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text(scanner.stats).monospacedDigit().font(.footnote)
                Spacer()
                Button { settings = true } label: { Image(systemName: "gearshape") }
                    .accessibilityLabel("設定")
            }
            GeometryReader { geometry in
                let width = min(geometry.size.width, geometry.size.height * scanner.aspect)
                let height = width / scanner.aspect
                ZStack {
                    Color.black
                    Preview(session: scanner.camera)
                    ZStack(alignment: .topLeading) {
                        ForEach(scanner.coins) { coin in
                            Rectangle().strokeBorder(.green, lineWidth: 2)
                                .overlay(alignment: .topLeading) {
                                    Text("\(denominations[coin.kind])円").font(.caption2.bold())
                                        .padding(2).background(.green).foregroundStyle(.black)
                                }
                                .frame(width: coin.rect.width * width, height: coin.rect.height * height)
                                .offset(x: coin.rect.minX * width, y: coin.rect.minY * height)
                        }
                    }
                    .frame(width: width, height: height, alignment: .topLeading)
                    .allowsHitTesting(false)
                    if scanner.loading { ProgressView("モデルを確認中").padding().background(.regularMaterial) }
                }.clipShape(RoundedRectangle(cornerRadius: 12))
            }
            if !scanner.message.isEmpty {
                Text(scanner.message).font(.caption).foregroundStyle(.red).lineLimit(4)
            }
            if scanner.supported.isEmpty {
                Button("ONNXモデルを選択") { importing = true }.disabled(scanner.loading)
            }
            let total = scanner.coins.reduce(0) { $0 + denominations[$1.kind] }
            Text("合計 \(total) 円").font(.largeTitle.bold()).monospacedDigit()
            if !scanner.backend.isEmpty { Text(scanner.backend).font(.caption).foregroundStyle(.secondary) }
        }
        .padding()
        .task { sync(); scanner.load(preferred: size); scanner.cameraActive(phase == .active) }
        .onChange(of: phase) { _, phase in scanner.cameraActive(phase == .active) }
        .onChange(of: size) { _, _ in sync() }
        .onChange(of: fps) { _, _ in sync() }
        .onChange(of: confidence) { _, _ in sync() }
        .onChange(of: iou) { _, _ in sync() }
        .onChange(of: stabilized) { _, _ in sync() }
        .onChange(of: scanner.supported) { _, _ in sync() }
        .sheet(isPresented: $settings) { settingsView }
        .fileImporter(isPresented: $importing, allowedContentTypes: [UTType(filenameExtension: "onnx") ?? .data]) {
            switch $0 {
            case .success(let url): scanner.load(url, preferred: size)
            case .failure(let error): scanner.message = error.localizedDescription
            }
        }
    }

    private var settingsView: some View {
        NavigationStack {
            Form {
                Button("モデルを変更") { settings = false; importing = true }.disabled(scanner.loading)
                Picker("入力解像度", selection: Binding(
                    get: { scanner.supported.isEmpty || scanner.supported.contains(size) ? size : scanner.supported[0] },
                    set: { size = $0 })) {
                    ForEach(scanner.supported.isEmpty ? resolutions : scanner.supported, id: \.self) {
                        Text("\($0)×\($0)").tag($0)
                    }
                }
                if scanner.supported.count == 1 {
                    Text("固定サイズのモデルです。別の解像度を使うには対応するONNXを選択します。")
                        .font(.caption)
                }
                Picker("推論FPS上限", selection: $fps) {
                    ForEach([5, 10, 15, 30], id: \.self) { Text("\($0) FPS").tag($0) }
                }
                LabeledContent("信頼度", value: confidence.formatted(.number.precision(.fractionLength(2))))
                Slider(value: $confidence, in: 0.05...0.95, step: 0.01).accessibilityLabel("信頼度")
                Toggle("検出結果を安定化", isOn: $stabilized)
                if stabilized {
                    LabeledContent("追跡のIoU", value: iou.formatted(.number.precision(.fractionLength(2))))
                    Slider(value: $iou, in: 0.15...0.85, step: 0.01).accessibilityLabel("追跡のIoU")
                }
                Button("すべてデフォルトに戻す", role: .destructive) { reset = true }
            }
            .navigationTitle("設定")
            .toolbar { Button("完了") { settings = false } }
            .confirmationDialog("設定を初期化しますか？", isPresented: $reset, titleVisibility: .visible) {
                Button("デフォルトに戻す", role: .destructive) {
                    size = 416; fps = 10; confidence = 0.25; iou = 0.45; stabilized = true
                }
            } message: { Text("インポート済みモデルは保持されます。") }
        }
    }

    private func sync() {
        scanner.configureInference(size: size, fps: fps, confidence: confidence, iou: iou, stabilized: stabilized)
    }
}
