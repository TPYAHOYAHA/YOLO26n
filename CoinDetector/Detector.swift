import AVFoundation
import CoreImage
import onnxruntime_objc

let denominations = [1, 5, 10, 50, 100, 500]
let resolutions = [320, 416, 640]

struct Coin: Identifiable {
    var id = UUID()
    var kind: Int
    var rect: CGRect
}

struct CoinError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

struct Stabilizer {
    private struct Track {
        var coin: Coin
        var hits: [Int]
        var seen: Double
        var confirmed = false
    }
    private var tracks: [Track] = []
    private var frame = 0

    mutating func update(_ coins: [Coin], time: Double, iou: Double) -> [Coin] {
        frame += 1
        tracks.removeAll { time - $0.seen >= 0.5 }
        var matched = Set<Int>()
        for coin in coins {
            let candidates = tracks.indices.filter {
                !matched.contains($0) && tracks[$0].coin.kind == coin.kind
            }
            if let index = candidates.max(by: {
                overlap(tracks[$0].coin.rect, coin.rect) < overlap(tracks[$1].coin.rect, coin.rect)
            }), overlap(tracks[index].coin.rect, coin.rect) >= iou {
                tracks[index].coin.rect = coin.rect
                tracks[index].seen = time
                tracks[index].hits.removeAll { $0 <= frame - 5 }
                tracks[index].hits.append(frame)
                if tracks[index].hits.count >= 3 { tracks[index].confirmed = true }
                matched.insert(index)
            } else {
                matched.insert(tracks.count)
                tracks.append(Track(coin: coin, hits: [frame], seen: time))
            }
        }
        return tracks.filter(\.confirmed).map(\.coin)
    }
}

private func overlap(_ a: CGRect, _ b: CGRect) -> Double {
    let intersection = a.intersection(b)
    guard !intersection.isNull else { return 0 }
    let area = intersection.width * intersection.height
    return Double(area / (a.width * a.height + b.width * b.height - area))
}

final class Detector {
    private let env: ORTEnv
    private var session: ORTSession
    private var url: URL
    private let context = CIContext()
    private var rgb: [UInt8] = []
    private var input = NSMutableData()
    private(set) var coreML = true
    private(set) var supported: [Int] = []

    func relocate(to url: URL) { self.url = url }

    init(url: URL, preferred: Int) throws {
        self.url = url
        let environment = try ORTEnv(loggingLevel: .warning)
        env = environment
        do {
            session = try Self.open(environment, url, coreML: true)
        } catch {
            session = try Self.open(environment, url, coreML: false)
            coreML = false
        }
        var lastError: Error?
        for size in [preferred] + resolutions.filter({ $0 != preferred }) {
            do {
                allocate(size)
                _ = try run(size: size)
                supported.append(size)
            } catch { lastError = error }
        }
        guard !supported.isEmpty else {
            throw CoinError("対応する入力は 320・416・640 の Float32 ONNX です。\n\(lastError?.localizedDescription ?? "")")
        }
        supported.sort()
    }

    private static func open(_ env: ORTEnv, _ url: URL, coreML: Bool) throws -> ORTSession {
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        if coreML {
            try options.appendCoreMLExecutionProvider(with: ORTCoreMLExecutionProviderOptions())
        }
        return try ORTSession(env: env, modelPath: url.path, sessionOptions: options)
    }

    private func allocate(_ size: Int) {
        let pixels = size * size
        if rgb.count != pixels * 4 {
            rgb = [UInt8](repeating: 0, count: pixels * 4)
            input = NSMutableData(length: pixels * 3 * MemoryLayout<Float>.size)!
        }
    }

    private func run(size: Int) throws -> [Float] {
        do {
            return try evaluate(size: size)
        } catch {
            guard coreML else { throw error }
            let previous = session
            session = try Self.open(env, url, coreML: false)
            do {
                let values = try evaluate(size: size)
                coreML = false
                return values
            } catch {
                session = previous
                throw error
            }
        }
    }

    private func evaluate(size: Int) throws -> [Float] {
        let names = try session.inputNames()
        let outputs = try session.outputNames()
        guard names.count == 1, outputs.count == 1 else {
            throw CoinError("入力と出力が各1つの検出モデルを選んでください。")
        }
        let tensor = try ORTValue(tensorData: input, elementType: .float,
                                  shape: [1, 3, NSNumber(value: size), NSNumber(value: size)])
        let values = try session.run(withInputs: [names[0]: tensor],
                                     outputNames: Set(outputs), runOptions: nil)
        guard let value = values[outputs[0]] else { throw CoinError("ONNX出力がありません。") }
        let info = try value.tensorTypeAndShapeInfo()
        let shape = info.shape.map(\.intValue)
        guard shape.count == 3, shape[0] == 1, (1...3000).contains(shape[1]),
              shape[2] == 6, info.elementType == .float else {
            throw CoinError("YOLO26の end-to-end 出力 [1, 件数, 6] が必要です。")
        }
        let data = try value.tensorData()
        let count = shape[1] * 6
        guard data.length == count * MemoryLayout<Float>.size else { throw CoinError("ONNX出力サイズが不正です。") }
        let pointer = data.bytes.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    func detect(_ buffer: CVPixelBuffer, size: Int, confidence: Float) throws -> [Coin] {
        allocate(size)
        let image = CIImage(cvPixelBuffer: buffer)
        let width = image.extent.width
        let height = image.extent.height
        let edge = CGFloat(size)
        let scale = min(edge / width, edge / height)
        let fittedWidth = (width * scale).rounded()
        let fittedHeight = (height * scale).rounded()
        let left = floor((edge - fittedWidth) / 2)
        let top = floor((edge - fittedHeight) / 2)
        let bottom = edge - fittedHeight - top
        let square = CGRect(x: 0, y: 0, width: edge, height: edge)
        let fitted = image.transformed(by: CGAffineTransform(scaleX: fittedWidth / width, y: fittedHeight / height))
            .transformed(by: CGAffineTransform(translationX: left, y: bottom))
        let gray = CGFloat(114) / 255
        let background = CIImage(color: CIColor(red: gray, green: gray, blue: gray)).cropped(to: square)
        rgb.withUnsafeMutableBytes { bytes in
            context.render(fitted.composited(over: background), toBitmap: bytes.baseAddress!,
                           rowBytes: size * 4, bounds: square, format: .RGBA8,
                           colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        }
        let pixels = size * size
        let float = input.mutableBytes.assumingMemoryBound(to: Float.self)
        for y in 0..<size {
            let source = (size - 1 - y) * size
            for x in 0..<size {
                let index = y * size + x
                let byte = (source + x) * 4
                float[index] = Float(rgb[byte]) / 255
                float[pixels + index] = Float(rgb[byte + 1]) / 255
                float[2 * pixels + index] = Float(rgb[byte + 2]) / 255
            }
        }
        let rows = try run(size: size)
        let candidates = stride(from: 0, to: rows.count, by: 6).compactMap { index -> (coin: Coin, score: Float)? in
            let row = rows[index..<(index + 6)]
            guard row.allSatisfy(\.isFinite), rows[index + 4] >= confidence,
                  let kind = Int(exactly: rows[index + 5]), denominations.indices.contains(kind)
            else { return nil }
            let x1 = (CGFloat(rows[index]) - left) / fittedWidth
            let y1 = (CGFloat(rows[index + 1]) - top) / fittedHeight
            let x2 = (CGFloat(rows[index + 2]) - left) / fittedWidth
            let y2 = (CGFloat(rows[index + 3]) - top) / fittedHeight
            guard x2 > x1, y2 > y1 else { return nil }
            let rect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
                .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            guard !rect.isNull, !rect.isEmpty else { return nil }
            return (Coin(kind: kind, rect: rect), rows[index + 4])
        }
        var unique: [Coin] = []
        for candidate in candidates.sorted(by: { $0.score > $1.score }) {
            if !unique.contains(where: { overlap($0.rect, candidate.coin.rect) >= 0.85 }) {
                unique.append(candidate.coin)
            }
        }
        return unique
    }
}
