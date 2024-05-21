export function createChunkDecoder() {
    const decoder = new TextDecoder();
    return (chunk: Uint8Array): string => decoder.decode(chunk, { stream: true });
}

export function createJsonChunkDecoder() {
    const decoder = new TextDecoder();
    return (chunk: Uint8Array): string => {
        const decoded_str = decoder.decode(chunk, { stream: true })
        try {
            return JSON.loads(decoded_str)
        } catch (e) {
            return {"message": decoded_str}
        }
    };
}
