export function createChunkDecoder() {
    const decoder = new TextDecoder();
    return (chunk: Uint8Array): string => decoder.decode(chunk, { stream: true });
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
