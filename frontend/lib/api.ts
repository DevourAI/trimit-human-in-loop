import {
  type StepOutputParams,
  type GetLatestStateParams,
  type ResetWorkflowParams,
  type RevertStepParams,
  type StepParams,
  type UploadVideoParams
} from './types'
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL

const fetcherWithParams = async (url, params) => {
  try {
    const res = await axios.get(url, { baseURL: API_URL, params })
    const toReturn = res.data;
    toReturn.headers = res.headers;
    return toReturn;
  } catch (error) {
    console.error('fetcherWithParams error', error);
    return { error }
  }
}

const fetcherWithParamsRaw = async (url, params, options = {}) => {
  try {
    const res = await axios.get(url, { baseURL: API_URL, params, ...options })
    return res;
  } catch (error) {
    console.error('fetcherWithParamsRaw error', error);
    return { error }
  }
}

const postFetcherWithData = async (url, data) => {
  const formData = new FormData();
  Object.keys(data).forEach(key => {
    if (Array.isArray(data[key])) {
      data[key].forEach(value => {
        formData.append(key, value);
      });
    } else {
      formData.append(key, data[key]);
    }
  });
  try {
    const res = await axios.post(url, formData, {
      baseURL: API_URL,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return res.data;
  } catch (error) {
    console.error('postFetcherWithData error', error);
    return { error }
  }
}

export async function resetWorkflow(params: ResetWorkflowParams): UserState {
  await fetcherWithParams('reset_workflow', params)
}
export async function revertStepInBackend(params: RevertStepParams): UserState {
  await fetcherWithParams('revert_workflow_step', params)
}

export async function getLatestState(params: GetLatestStateParams): UserState {
  if (params.user_email === '') return {}
  params.with_output = params.with_output || true
  params.wait_until_done_running = params.wait_until_done_running || false
  params.block_until = params.block_until || false
  params.timeout = params.timeout || 5
  const data = await fetcherWithParams('get_latest_state', params)
  if (data && data.error) {
    console.error(data.error)
  } else if (data) {
    return data;
  }
  return {}
}

export async function getStepOutput(params: StepOutputParams) {
  if (params.user_email === '') return {}
  params.latest_retry = params.latest_retry || true
  params.step_keys = params.step_key
  delete params.step_key
  const data = await fetcherWithParams('get_step_outputs', params)
  if (data && data.error) {
    console.error(data.error)
  } else if (data && data.outputs) {
    return data.outputs[0];
  }
  return {}
}

export async function step(params: StepParams, streamReaderCallback) {
  const url = new URL(`${API_URL}/step`)
  Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
  try {
    const res = await fetch(
      url.toString(),
      {
        method: "GET",
      }
    ).catch((err) => {
      throw err;
    });
    if (!res.ok) {
        throw new Error((await res.text()) || "Failed to fetch the chat response.");
    }

    if (!res.body) {
        throw new Error("The response body is empty.");
    }
    const reader = res.body.getReader();
    streamReaderCallback(reader);
  } catch (err: unknown) {
    console.error(err);
  }
}
export async function uploadVideo(params: UploadVideoParams) {
  if (params.userEmail === '') return {}
  if (!params.videoFile) return {}
  const data = {
    user_email: params.userEmail,
    files: [params.videoFile],
    timeline_name: params.timelineName,
    high_res_user_file_paths: [params.videoFile.name],
    reprocess: true,
    use_existing_output: true,
    overwrite: true
  }

  const respData = await postFetcherWithData('upload', data)

  if (respData && respData.result && respData.result.error) {
    console.error(respData)
  } else if (respData && respData.videoHashes && respData.videoHashes.length) {
    return {"videoHash": respData.videoHashes[0], "callId": respData.processing_call_id}
  }
  return respData
}

function remoteVideoStreamURLForPath(path: string) {
  return `${API_URL}/video?video_path=${path}`
}
export async function getUploadedVideos(params: GetUploadedVideoParams) {
  if (params.user_email === '') return {}
  const respData = await fetcherWithParams('uploaded_videos', params)

  if (respData && respData.error) {
    console.error(respData)
  } else if (respData && respData.length > 0) {
    return respData.map(video => {
      return {
        filename: video.filename,
        hash: video.video_hash,
        remoteUrl: remoteVideoStreamURLForPath(video.path)
      }
    })
  }
  return respData
}


const endpointForFileType = {
  'timeline': 'download_timeline',
  'video': 'video',
  'transcript_text': 'download_transcript_text',
  'soundbites_text': 'download_soundbites_text',
}
export async function downloadFile(params: DownloadFileParams) {
  if (params.user_email === '') return {}
  params.stream = false
  const filetype = params.filetype
  delete params.filetype
  const endpoint = endpointForFileType[filetype]
  const response = await fetcherWithParamsRaw(endpoint, params, { responseType: 'blob' })
  if (response.error) {
    console.error(response.error)
    return
  }

  if (typeof response.data !== 'string' && response.data.error) {
    console.error(response.data.error)
    return
  }
  const blob = new Blob([response.data], { type: response.headers['content-type'] });
  const contentDisposition = response.headers['content-disposition'];
  // TODO add user file path to default filename
  let filename = `trimit_${filetype}_for_user_${params.user_email}.mp4`; // Default filename
  if (contentDisposition) {
    const match = contentDisposition.match(/filename="(.+)"/);
    if (match.length > 1) {
      filename = match[1];
    }
  }
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
};
export async function downloadVideo(params: DownloadFileParams) {
  await downloadFile({ ...params, filetype: 'video' })
};
export async function downloadTimeline(params: DownloadTimelineParams) {
  await downloadFile({ ...params, filetype: 'timeline' })
};
export async function downloadTranscriptText(params: DownloadFileParams) {
  await downloadFile({ ...params, filetype: 'transcript_text' })
};
export async function downloadSoundbitesText(params: DownloadTimelineParams) {
  await downloadFile({ ...params, filetype: 'soundbites_text' })
};
