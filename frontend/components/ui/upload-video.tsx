'use client';
import React, { ChangeEvent, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { uploadVideo as uploadVideoAPI } from '@/lib/api';
import { UploadVideoParams } from '@/lib/types';

interface UploadVideoProps {
  userEmail: string;
  setVideoDetails: (hash: string, filename: string, title: string) => void;
  videoProcessingStatuses: { [key: string]: { status: string } };
  setUploadingVideo: (hash: string | null) => void;
  setVideoProcessingStatuses: (statuses: {
    [key: string]: { status: string };
  }) => void;
}

export default function UploadVideo({
  userEmail,
  setUploadingVideo,
  setVideoDetails,
  videoProcessingStatuses,
  setVideoProcessingStatuses,
}: UploadVideoProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fileToUpload, setFileToUpload] = useState<File | null>(null);
  const [weblinkToUpload, setWeblinkToUpload] = useState<string | null>(null);

  const handleVideoChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setFileToUpload(file);
  };
  const handleWeblinkChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const text = event.target.value;
    setWeblinkToUpload(text);
  };
  const uploadWeblink = async () => {
    if (!weblinkToUpload) {
      return;
    }
    const uploadVideoParams = {
      weblink: weblinkToUpload,
      userEmail,
      timelineName: 'timelineName', // Replace with actual timeline name if needed
    };
    const urlSearchParams = new URLSearchParams();
    urlSearchParams.append('weblink', weblinkToUpload);
    const url = `/api/youtube?${urlSearchParams.toString()}`;
    const resp = await fetch(url);
    const respData = await resp.json();
    console.log('api/youtube respData', respData);
    const title = respData.title || weblinkToUpload;

    console.log('setting uploading video to', title);
    setUploadingVideo({ title, filename_or_weblink: weblinkToUpload });
    setVideoDetails('', weblinkToUpload, title);
    await upload(uploadVideoParams as UploadVideoParams);
  };
  const uploadFile = async () => {
    if (!fileToUpload) {
      return;
    }
    const uploadVideoParams = {
      videoFile: fileToUpload,
      userEmail,
      timelineName: 'timelineName', // Replace with actual timeline name if needed
    };
    console.log('setting uploading video to', fileToUpload.name);
    setUploadingVideo({
      filename_or_weblink: fileToUpload.name,
      title: fileToUpload.name,
    });
    setVideoDetails('', fileToUpload.name);
    await upload(uploadVideoParams as UploadVideoParams);
  };
  const upload = async (params: UploadVideoParams) => {
    setUploadError(null); // Clear any previous errors
    setIsUploading(true);
    try {
      const respData = await uploadVideoAPI(params);
      if (
        respData &&
        respData.callId &&
        respData.videoHash &&
        respData.filename
      ) {
        const newEntries = {
          [respData.videoHash]: {
            callId: respData.callId,
            status: 'pending',
          },
        };
        setVideoProcessingStatuses({
          ...videoProcessingStatuses,
          ...newEntries,
        });
        setVideoDetails(respData.videoHash, respData.filename);
      } else {
        setUploadError('Failed to upload video. Please try again.');
        console.error('Error uploading video', respData);
      }
    } catch (error) {
      console.error('Error uploading video', error);
      setUploadError('Failed to upload video. Please try again.');
    }
    setUploadingVideo(null);
    setIsUploading(false);
  };

  return (
    <div className="grid w-full items-center gap-2 text-sm">
      <div className="flex flex-col items-start gap-2">
        <div className="flex items-center gap-2">
          <Label htmlFor="video" className="cursor-pointer">
            Upload a video
          </Label>
          <Input
            id="video"
            onChange={handleVideoChange}
            accept="video/*"
            type="file"
            className="mt-1 cursor-pointer text-muted-foreground max-w-sm"
          />
          <Button onClick={uploadFile} disabled={!fileToUpload || isUploading}>
            Upload
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Label htmlFor="link-str" className="cursor-pointer">
            Or, paste a Youtube link
          </Label>
          <Input
            id="link-str"
            onChange={handleWeblinkChange}
            type="text"
            className="mt-1 cursor-pointer text-muted-foreground max-w-sm"
          />
          <Button
            onClick={uploadWeblink}
            disabled={!weblinkToUpload || isUploading}
          >
            Upload
          </Button>
        </div>
      </div>

      {isUploading && (
        <div className="flex items-center gap-1">
          <LoadingSpinner size="small" />
          <div className="text-muted-foreground">Uploading...</div>
        </div>
      )}
      {uploadError && (
        <div className="ml-2 text-red-600">
          <p>{uploadError}</p>
        </div>
      )}
    </div>
  );
}
