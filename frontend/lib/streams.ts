export function createChunkDecoder() {
    const decoder = new TextDecoder();
    return (chunk: Uint8Array): string => decoder.decode(chunk, { stream: true });
}


export async function decodeStreamAsJSON(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    callback: (data: any) => void
) {
    let buffer = '';
    const decoder = new TextDecoder();
    let lastValue = null;

    async function read() {
        const { done, value } = await reader.read();
        if (done) {
            if (buffer.length > 0) {
                lastValue = processChunk(buffer);
            }
            return done;
        }
        const chunk = new TextDecoder("utf-8").decode(value);
        buffer += chunk;
        const parts = buffer.split("\n");
        for (let i = 0; i < parts.length - 1; i++) {
            lastValue = processChunk(parts[i]);
        }
        buffer = parts[parts.length - 1];
    }
    function processChunk(text) {
        if (text) {
            let valueDecoded = null;
            try {
                valueDecoded = JSON.parse(text);
            } catch (e) {
                console.error(e);
                return null
            }
            if (valueDecoded && !valueDecoded.is_last && valueDecoded.result) {
                callback(valueDecoded.result);
            } else if (valueDecoded && valueDecoded.is_last && valueDecoded.result) {
                return valueDecoded.result;
            } else if (valueDecoded && valueDecoded.is_last) {
                return valueDecoded;
            } else if (valueDecoded && valueDecoded.message) {
                callback(valueDecoded);
            } else {
                console.log("No message or result found in message", valueDecoded)
            }
        }
        return null;
    }
    while (true) {
        const done = await read();
        if (done) {
            break;
        }
    }
    if (!lastValue) {
        console.error("No last value found in stream");
        console.error("Buffer: ", buffer);
    }
    return lastValue;
}
export function createJsonChunkDecoder() {
    const decoder = new TextDecoder();
    return (chunk: Uint8Array): string => {
        const decoded_str = decoder.decode(chunk, { stream: true })
        const parts = decoded_str.split('\n')
        try {
            return parts.map((part) => JSON.parse(part))
        } catch (e) {
            console.error(e)
            console.log(decoded_str)
            return []
        }
    };
}
