import {
  CutTranscriptLinearWorkflowStreamingOutput,
  GetLatestState,
} from '@/gen/openapi/api';

const MAX_READ_FAILURES = 10;

// Function to create a chunk decoder using TextDecoder
export function createChunkDecoder() {
  const decoder = new TextDecoder();
  return (chunk: Uint8Array): string => decoder.decode(chunk, { stream: true });
}

// Function to decode a stream as JSON and process the data using a callback
export async function decodeStreamAsJSON(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  callback: (output: CutTranscriptLinearWorkflowStreamingOutput) => void
): Promise<GetLatestState | null> {
  let buffer = ''; // Buffer to accumulate chunks of data
  let lastValue = null; // Variable to store the last processed value
  // Function to process a chunk of text and parse it as JSON
  function processChunk(text: string) {
    if (text) {
      let valueDecoded = null;
      try {
        valueDecoded = JSON.parse(text);
      } catch (e) {
        return [null, false];
      }
      if (valueDecoded && valueDecoded.final_state) {
        return [valueDecoded.final_state, true];
      } else if (valueDecoded) {
        console.log('valueDecoded, passing to callback', valueDecoded);
        callback(valueDecoded);
      } else {
        console.log('Could not parse message', valueDecoded);
      }
    }
    return [null, true];
  }

  // Function to read data from the stream
  async function read() {
    let done = false;
    let value = new Uint8Array(0);
    try {
      const res = await reader.read();
      done = res.done;
      value = res.value ? res.value : value;
    } catch (error) {
      console.error(error);
      return { done: false, success: false };
    }
    if (done) {
      if (buffer.length > 0) {
        [buffer, lastValue] = splitAndProcessBuffer(buffer);
      }
      return { done, success: true };
    }
    const chunk = new TextDecoder('utf-8').decode(value);
    buffer += chunk;
    [buffer, lastValue] = splitAndProcessBuffer(buffer);
    return { done: false, success: true };
  }
  function splitAndProcessBuffer(buffer: string) {
    const parts = buffer.split(/(?<=})(?={)/);
    let lastValue = null;
    let parsed = false;
    const n = parts.length > 1 ? parts.length - 1 : parts.length;
    for (let i = 0; i < n; i++) {
      [lastValue, parsed] = processChunk(parts[i]);
    }
    if (parts.length > 1) {
      buffer = parts[parts.length - 1];
    } else if (parsed) {
      buffer = '';
    }
    return [buffer, lastValue];
  }
  let nFailures = 0; // Counter for the number of read failures
  while (true) {
    const result = await read();
    if (result.success && result.done) {
      break;
    } else if (!result.success) {
      console.error('read failed', result);
      nFailures++;
      if (nFailures >= MAX_READ_FAILURES) {
        console.error(
          `max read failures reached (${MAX_READ_FAILURES}), breaking`
        );
        break;
      }
    }
  }
  if (!lastValue) {
    console.error('No last value found in stream');
    console.error('Buffer: ', buffer);
  }
  return lastValue;
}
