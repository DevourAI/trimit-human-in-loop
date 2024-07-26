'use client';
import React, { ChangeEvent, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { uploadVideo as uploadVideoAPI } from '@/lib/api';

interface UploadVideoProps {
  userEmail: string;
  setVideoDetails: (hash: string, filename: string) => void;
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

  const handleVideoChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setFileToUpload(file);
  };
  const upload = async () => {
    if (fileToUpload) {
      setUploadError(null); // Clear any previous errors
      setIsUploading(true);
      console.log('setting uploading video to', fileToUpload.name);
      setUploadingVideo(fileToUpload.name);
      setVideoDetails('', fileToUpload.name);
      try {
        const respData = await uploadVideoAPI({
          videoFile: fileToUpload,
          userEmail,
          timelineName: 'timelineName', // Replace with actual timeline name if needed
        });
        if (respData && respData.processing_call_id) {
          const newEntries = {
            [respData.video_hashes[0]]: {
              callId: respData.processing_call_id,
              status: 'pending',
            },
          };
          setVideoProcessingStatuses({
            ...videoProcessingStatuses,
            ...newEntries,
          });
          setVideoDetails(respData.video_hashes[0], fileToUpload.name);
        }
      } catch (error) {
        console.error('Error uploading video', error);
        setUploadError('Failed to upload video. Please try again.');
      }
      setUploadingVideo(null);
      setIsUploading(false);
    }
  };

  return (
    <div className="grid w-full items-center gap-1 text-sm">
      <Label htmlFor="video" className="cursor-pointer">
        Upload a video
      </Label>
      <div className="flex items-center gap-2">
        <Input
          id="video"
          onChange={handleVideoChange}
          accept="video/*"
          type="file"
          className="mt-1 cursor-pointer text-muted-foreground max-w-sm"
        />
        <Button onClick={upload} disabled={!fileToUpload || isUploading}>
          Upload
        </Button>
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
    </div>
  );
}
