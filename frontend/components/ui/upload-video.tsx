"use client";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { ReloadIcon } from "@radix-ui/react-icons";
import React, { useState, ChangeEvent } from "react";
import { uploadVideo as uploadVideoAPI } from "@/lib/api";
import { LoadingSpinner } from "@/components/ui/loading-spinner";

interface UploadVideoProps {
  userEmail: string;
  setVideoHash: (hash: string) => void;
  videoProcessingStatuses: { [key: string]: { status: string } };
  setVideoProcessingStatuses: (statuses: {
    [key: string]: { status: string };
  }) => void;
}

export default function UploadVideo({
  userEmail,
  setVideoHash,
  videoProcessingStatuses,
  setVideoProcessingStatuses,
}: UploadVideoProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);

  const handleVideoChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    if (file) {
      setSelectedVideo(file);
      setUploadError(null); // Clear any previous errors
      setIsUploading(true);
      try {
        const respData = await uploadVideoAPI({
          videoFile: file,
          userEmail,
          timelineName: "timelineName", // Replace with actual timeline name if needed
        });
        if (respData && respData.processing_call_id) {
          const newEntries = {
            [respData.video_hashes[0]]: {
              callId: respData.processing_call_id,
              status: "pending",
            },
          };
          setVideoProcessingStatuses({
            ...videoProcessingStatuses,
            ...newEntries,
          });
          setVideoHash(respData.video_hashes[0]);
        }
      } catch (error) {
        console.error("Error uploading video", error);
        setUploadError("Failed to upload video. Please try again.");
      }
      setIsUploading(false);
    }
  };

  return (
    <div className="grid w-full items-center gap-1 max-w-sm text-sm">
      <Label htmlFor="video" className="cursor-pointer">
        Upload a video
      </Label>
      <div className="flex items-center gap-2">
        <Input
          id="video"
          onChange={handleVideoChange}
          accept="video/*"
          type="file"
          className="mt-1 cursor-pointer text-muted-foreground"
        />
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
