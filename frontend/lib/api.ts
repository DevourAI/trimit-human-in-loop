import axios, { AxiosResponse } from 'axios';

import {
  CheckFunctionCallResults,
  CutTranscriptLinearWorkflowStepOutput,
  UploadedVideo,
  UploadVideo,
  UploadVideoApi,
  VideosApi,
} from '@/gen/openapi/api';
import {
  DownloadFileParams,
  FrontendWorkflowState,
  GetLatestStateParams,
  GetUploadedVideoParams,
  ResetWorkflowParams,
  RevertStepParams,
  RevertStepToParams,
  StepOutputParams,
  StepParams,
  UploadVideoParams,
  UserState,
} from '@/lib/types';

let API_URL = process.env.NEXT_PUBLIC_API_BASE_URL_REMOTE;
if (process.env.BACKEND === 'local') {
  API_URL = process.env.NEXT_PUBLIC_API_BASE_URL_LOCAL;
}

const uploadVideosApi = new UploadVideoApi(undefined, API_URL);
const videosApi = new VideosApi(undefined, API_URL);

const fetcherWithParams = async (
  url: string,
  params: Record<string, unknown>
): Promise<unknown> => {
  try {
    const res: AxiosResponse = await axios.get(url, {
      baseURL: API_URL,
      params,
    });
    const toReturn = res.data;
    if (toReturn !== null && typeof toReturn === 'object') {
      toReturn.headers = res.headers;
    }
    return toReturn;
  } catch (error) {
    console.error('fetcherWithParams error', error);
    return { error };
  }
};

const fetcherWithParamsRaw = async (
  url: string,
  params: Record<string, unknown>,
  options: Record<string, unknown> = {}
): Promise<unknown> => {
  try {
    const res: AxiosResponse = await axios.get(url, {
      baseURL: API_URL,
      params,
      ...options,
    });
    return res;
  } catch (error) {
    console.error('fetcherWithParamsRaw error', error);
    return { error };
  }
};

const postFetcherWithData = async (
  url: string,
  data: Record<string, unknown>
): Promise<unknown> => {
  const formData = new FormData();
  Object.keys(data).forEach((key) => {
    if (Array.isArray(data[key])) {
      data[key].forEach((value) => {
        formData.append(key, value);
      });
    } else {
      formData.append(key, data[key]);
    }
  });
  try {
    const res: AxiosResponse = await axios.post(url, formData, {
      baseURL: API_URL,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  } catch (error) {
    console.error('postFetcherWithData error', error);
    return { error };
  }
};

export async function resetWorkflow(
  params: ResetWorkflowParams
): Promise<UserState | boolean> {
  const result = await fetcherWithParams('reset_workflow', params);
  if (result && result.success === false) {
    return false;
  }
  return true;
}

export async function revertStepInBackend(
  params: RevertStepParams
): Promise<boolean> {
  const result = await fetcherWithParams('revert_workflow_step', params);
  if (result && result.success === false) {
    return false;
  }
  return true;
}

export async function revertStepToInBackend(
  params: RevertStepToParams
): Promise<boolean> {
  const result = await fetcherWithParams('revert_workflow_step_to', params);
  console.log('revertStepToInBackend result', result);
  if (result && result.error !== undefined) {
    console.log('returning false');
    return false;
  }
  console.log('returning true');
  return true;
}

export async function getLatestState(
  params: GetLatestStateParams
): Promise<FrontendWorkflowState> {
  if (
    !params.user_email ||
    !params.video_hash ||
    !params.length_seconds ||
    !params.timeline_name
  )
    return {};
  params.with_output = params.with_output ?? true;
  params.wait_until_done_running = params.wait_until_done_running ?? false;
  params.block_until = params.block_until ?? false;
  params.timeout = params.timeout ?? 5;
  const data = await fetcherWithParams('get_latest_state', params);
  if (data && data.error) {
    console.error(data.error);
    return { success: false, error: data.error };
  } else if (data) {
    return data as FrontendWorkflowState;
  }
  return {} as FrontendWorkflowState;
}

export async function getStepOutput(
  params: StepOutputParams
): Promise<CutTranscriptLinearWorkflowStepOutput | null> {
  if (params.user_email === '' || !params.video_hash) return null;
  params.latest_retry = params.latest_retry ?? true;
  params.step_names = params.step_name;
  delete params.step_name;
  const data = await fetcherWithParams('get_step_outputs', params);
  if (data && data.error) {
    console.error(data.error);
  } else if (data && data.outputs) {
    return data.outputs[0];
  }
  return null;
}

export async function step(
  params: StepParams,
  streamReaderCallback: (reader: ReadableStreamDefaultReader) => void
): Promise<void> {
  const url = new URL(`${API_URL}/step`);
  Object.keys(params).forEach((key) => {
    if (params[key] !== undefined && params[key] !== null) {
      url.searchParams.append(key, params[key]);
    }
  });
  try {
    const res = await fetch(url.toString(), {
      method: 'GET',
    }).catch((err) => {
      throw err;
    });
    if (!res.ok) {
      throw new Error(
        (await res.text()) || 'Failed to fetch the chat response.'
      );
    }

    if (!res.body) {
      throw new Error('The response body is empty.');
    }
    const reader = res.body.getReader();
    streamReaderCallback(reader);
  } catch (err: unknown) {
    console.error(err);
  }
}
export async function uploadVideo(params: UploadVideoParams): Promise<unknown> {
  if (params.userEmail === '' || !params.videoFile) return {};
  let data: UploadVideo;
  try {
    const response = await uploadVideosApi.uploadMultipleFilesUploadPost(
      [params.videoFile],
      [params.videoFile.name],
      params.timelineName,
      params.userEmail,
      true,
      false,
      true
    );
    // TODO test this- do we need to take the data field or it already extracted?
    data = response.data;
  } catch (error) {
    console.error(error);
    return;
  }

  if (data.result === 'error') {
    console.error(data.messages);
  } else if (data.video_hashes && data.video_hashes.length) {
    return {
      videoHash: data.video_hashes[0],
      callId: data.processing_call_id,
    };
  }
  return data;
}

export async function getVideoProcessingStatuses(
  userEmail: string,
  options: { timeout?: number } = { timeout: 0 }
): Promise<unknown> {
  if (!userEmail) {
    return { result: 'error', message: 'No userEmail provided' };
  }
  const respData = await fetcherWithParams('get_video_processing_status', {
    user_email: userEmail,
    ...options,
  });
  if (respData && respData.result) {
    return respData;
  }
  return { result: 'error', message: 'No result found' };
}

export async function getFunctionCallResults(
  callIds: string[],
  options: { timeout?: number } = { timeout: 0 }
): Promise<unknown> {
  if (!callIds || callIds.length === 0) {
    return { result: 'error', message: 'No callIds provided' };
  }
  const respData = (await fetcherWithParams('check_function_call_results', {
    modal_call_ids: callIds,
    ...options,
  })) as CheckFunctionCallResults;
  if (respData && respData.statuses) {
    return respData.statuses;
  }
  return { result: 'error', message: 'No result found' };
}

function remoteVideoStreamURLForPath(path: string): string {
  return `${API_URL}/video?video_path=${path}`;
}

export async function getUploadedVideos(
  params: GetUploadedVideoParams
): Promise<unknown> {
  if (params.user_email === '') return {};

  console.log(params.user_email);
  const respData = await videosApi.uploadedVideosUploadedVideosGet(
    params.user_email
  );
  console.log('upload video resp data', respData);

  const data = respData.data;
  if (respData && respData.error) {
    console.error(respData);
  } else if (data && data.length > 0) {
    return data.map((video: UploadedVideo) => {
      return {
        filename: video.filename,
        hash: video.video_hash,
        remoteUrl: remoteVideoStreamURLForPath(video.path),
      };
    });
  }
  return respData;
}

const endpointForFileType: Record<string, string> = {
  timeline: 'download_timeline',
  video: 'video',
  transcript_text: 'download_transcript_text',
  soundbites_text: 'download_soundbites_text',
};

export async function downloadFile(params: DownloadFileParams): Promise<void> {
  if (params.user_email === '' || !params.video_hash) return;
  params.stream = false;
  const filetype = params.filetype;
  delete params.filetype;
  const endpoint = endpointForFileType[filetype];
  const response = await fetcherWithParamsRaw(endpoint, params, {
    responseType: 'blob',
  });
  if (response.error) {
    console.error(response.error);
    return;
  }

  if (typeof response.data !== 'string' && response.data.error) {
    console.error(response.data.error);
    return;
  }
  const blob = new Blob([response.data], {
    type: response.headers['content-type'],
  });
  const contentDisposition = response.headers['content-disposition'];
  // TODO add user file path to default filename
  let filename = `trimit_${filetype}_for_user_${params.user_email}.mp4`; // Default filename
  if (contentDisposition) {
    const match = contentDisposition.match(/filename="(.+)"/);
    if (match && match.length > 1) {
      filename = match[1];
    }
  }
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
}

export async function downloadVideo(params: DownloadFileParams): Promise<void> {
  await downloadFile({ ...params, filetype: 'video' });
}

export async function downloadTimeline(
  params: DownloadFileParams
): Promise<void> {
  await downloadFile({ ...params, filetype: 'timeline' });
}

export async function downloadTranscriptText(
  params: DownloadFileParams
): Promise<void> {
  await downloadFile({ ...params, filetype: 'transcript_text' });
}

export async function downloadSoundbitesText(
  params: DownloadFileParams
): Promise<void> {
  await downloadFile({ ...params, filetype: 'soundbites_text' });
}
