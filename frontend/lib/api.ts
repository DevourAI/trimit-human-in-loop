import axios, { AxiosResponse } from 'axios';

import {
  CheckFunctionCallResults,
  CutTranscriptLinearWorkflowStepOutput,
  ExportResults,
  FrontendWorkflowProjection,
  RunInput,
  UploadedVideo,
  UploadVideo,
  UploadVideoApi,
  VideosApi,
} from '@/gen/openapi/api';
import {
  DownloadFileParams,
  FrontendWorkflowState,
  GetLatestExportResultParams,
  GetUploadedVideoParams,
  ListWorkflowParams,
  RedoExportResultParams,
  RevertStepParams,
  RevertStepToParams,
  StepData,
  StepOutputParams,
  StepQueryParams,
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

// const postJSONFetcherWithStreamingResponse = async (
// url: string,
// options: {
// data: Record<string, unknown>,
// params?: Record<string, unknown>,
// streamReaderCallback?: (reader: ReadableStreamDefaultReader) => void
// }
// ): Promise<unknown> => {
// let res: AxiosResponse;
// const axiosConfig = {
// url,
// method: 'post',
// baseURL: API_URL,
// data: options.data,
// params: options.params,
// };
// console.log('axiosConfig', axiosConfig);
// console.log('options', options);
// try {
// res = await axios(axiosConfig);
// } catch (error) {
// console.error('postFetcherWithData error', error);
// return { error };
// }
// if (!res.body) {
// const errorMsg ='The response body is empty.';
// console.error(errorMsg);
// return {error: errorMsg}
// }
// if (options.streamReaderCallback) {
// const reader = res.body.getReader();
// options.streamReaderCallback(reader);
// }
// };
const postJSONFetcherWithStreamingResponse = async (
  url: string,
  options: {
    data: Record<string, any>;
    params?: Record<string, any>;
    streamReaderCallback?: (reader: ReadableStreamDefaultReader) => void;
  }
): Promise<unknown> => {
  // Prepare URL with query parameters if any
  const urlWithParams = new URL(url);
  if (options.params) {
    Object.keys(options.params).forEach((key) =>
      urlWithParams.searchParams.append(key, options.params[key])
    );
  }

  // Prepare fetch options
  const fetchOptions = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(options.data),
  };

  try {
    const response = await fetch(urlWithParams.toString(), fetchOptions);

    // Check if the response is ok (status in the range 200-299)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // If a streamReaderCallback is provided and the body is usable as a stream
    if (options.streamReaderCallback && response.body) {
      const reader = response.body.getReader();
      options.streamReaderCallback(reader);
      return;
    }

    // If streaming is not required, simply return the parsed JSON
    return await response.json();
  } catch (error) {
    console.error('postFetcherWithData error', error);
    return { error };
  }
};

export async function createNewWorkflow(
  params: CreateNewWorkflowParams
): Promise<string> {
  return await postFetcherWithData('workflows/new', params);
}

export async function resetWorkflow(
  workflowId: string
): Promise<UserState | boolean> {
  const result = await fetcherWithParams('reset_workflow', {
    workflow_id: workflowId,
  });
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
  if (result && result.error !== undefined) {
    return false;
  }
  return true;
}

export async function getLatestState(
  workflowId: string
): Promise<FrontendWorkflowState> {
  if (!workflowId) return {};
  const params = { workflow_id: workflowId };
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
  workflowId: string,
  data: StepData,
  streamReaderCallback: (reader: ReadableStreamDefaultReader) => void
): Promise<void> {
  const params = { workflow_id: workflowId };
  await postJSONFetcherWithStreamingResponse(`${API_URL}/step`, {
    params,
    data,
    streamReaderCallback,
  });
}

export interface LocalRunInput extends RunInput {
  project_id: string;
  project_name: string;
  workflow_id: string;
}
export async function run(
  data: LocalRunInput,
  streamReaderCallback: (reader: ReadableStreamDefaultReader) => void
): Promise<void> {
  const params = {
    project_id: data.project_id,
    project_name: data.project_name,
    workflow_id: data.workflow_id,
    user_email: data.user_email,
  };
  await postJSONFetcherWithStreamingResponse(`${API_URL}/run`, {
    params,
    data,
    streamReaderCallback,
  });
}

export async function listWorkflows(
  params: ListWorkflowParams
): Promise<FrontendWorkflowProjection[]> {
  const data = await fetcherWithParams('workflows', params);
  if (data && data.error) {
    console.error(data.error);
  } else if (data) {
    return data;
  }
  return null;
}
export async function getWorkflowDetails(
  workflowId: string
): Promise<FrontendWorkflowProjection> {
  const data = await fetcherWithParams('workflow', { workflow_id: workflowId });
  if (data && data.error) {
    console.error(data.error);
  } else if (data) {
    return data;
  }
  return null;
}
export async function checkWorkflowExists(
  queryParams: StepQueryParams
): Promise<boolean> {
  const data = await fetcherWithParams('workflow_exists', queryParams);
  if (data && data.error) {
    console.error(data.error);
  } else if (data && data.exists) {
    return data.exists;
  }
  return false;
}

export interface UploadVideoResponse {
  videoHash?: string;
  filename?: string;
  callId?: string | null;
}

export async function uploadVideo(
  params: UploadVideoParams
): Promise<UploadVideoResponse> {
  if (params.userEmail === '' || (!params.videoFile && !params.weblink))
    return {};
  let data: UploadVideo;
  try {
    const videoFiles = params.videoFile ? [params.videoFile] : [];
    const weblinks = params.weblink ? [params.weblink] : [];
    const videoFileNames = videoFiles.map((file) => file.name);
    const response = await uploadVideosApi.uploadMultipleFilesUploadPost(
      params.timelineName,
      params.userEmail,
      videoFiles,
      videoFileNames,
      weblinks,
      true,
      false,
      true
    );
    // TODO test this- do we need to take the data field or it already extracted?
    data = response.data;
  } catch (error) {
    console.error(error);
    return {};
  }

  if (data.result === 'error') {
    console.error(data.messages);
  } else if (data.video_hashes && data.video_hashes.length) {
    return {
      videoHash: data.video_hashes[0],
      filename: data.filenames ? data.filenames[0] : undefined,
      title: data.titles ? data.titles[0] : undefined,
      callId: data.processing_call_id,
    };
  }
  return {};
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
  if (respData) {
    return { result: respData };
  }
  console.log('getVideoProcessingStatuses', respData);
  return { result: 'error', message: 'No result found' };
}

export async function getFunctionCallResults(
  callIds: string[],
  options: { timeout?: number } = { timeout: 0 }
): Promise<unknown> {
  const callIdsFiltered = callIds.filter((callId) => {
    return callId !== undefined && callId !== null;
  });
  if (callIdsFiltered.length === 0) {
    return { result: 'error', message: 'No callIds provided' };
  }
  const respData = (await fetcherWithParams('check_function_call_results', {
    modal_call_ids: encodeURIComponent(callIds.join(',')),
    ...options,
  })) as CheckFunctionCallResults;

  if (respData && respData.statuses) {
    return respData.statuses;
  }
  return { result: 'error', message: 'No result found' };
}

// export function remoteVideoStreamURLForPath(path) {
// if (!path.startsWith('http')) {
// return `${API_URL}/video?video_path=${encodeURIComponent(path)}`;
// } else {
// return path;
// }
// }

export async function getUploadedVideos(
  params: GetUploadedVideoParams
): Promise<UploadedVideo[]> {
  if (params.user_email === '') return {};

  const respData = await videosApi.uploadedVideosUploadedVideosGet(
    params.user_email
  );

  const data = respData.data;
  if (respData && respData.error) {
    console.error(respData);
  } else if (data && data.length > 0) {
    return data;
  }
  return [];
}

export async function redoExportResults(
  params: RedoExportResultParams
): Promise<string> {
  const workflowId = params.workflow_id;
  const response = await postFetcherWithData(
    `redo_export_results?workflow_id=${workflowId}`,
    { step_name: params.step_name }
  );
  if (response) {
    return response.call_id;
  }
  return '';
}
export async function getLatestExportResults(
  params: GetLatestExportResultParams
): Promise<ExportResults> {
  const respData = (await fetcherWithParams('get_latest_export_results', {
    workflow_id: params.workflow_id,
    step_name: params.step_name,
  })) as any;
  delete respData.headers;

  return respData as ExportResults;
}

const endpointForFileType: Record<string, string> = {
  timeline: 'download_timeline',
  video: 'video',
  transcript_text: 'download_transcript_text',
  soundbites_text: 'download_soundbites_text',
  soundbites_timeline: 'download_soundbites_timeline',
};

export async function downloadFile(params: DownloadFileParams): Promise<void> {
  if (params.workflow_id === '' || !params.workflow_id) return;
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
  let filename = `trimit_${filetype}_for_user_email_${params.user_email}.mp4`; // Default filename
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

export async function downloadSoundbitesTimeline(
  params: DownloadFileParams
): Promise<void> {
  await downloadFile({ ...params, filetype: 'soundbites_timeline' });
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
