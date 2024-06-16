import React, { useEffect, useState, useRef } from 'react';
import type { GetUploadedVideoParams, Video } from '@/lib/types';
import { getUploadedVideos, getVideoProcessingStatuses } from '@/lib/api';
import UploadVideo from '@/components/ui/upload-video';
import VideoTable from '@/components/ui/video-table';
import { useUser } from '@/contexts/user-context';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
const POLL_INTERVAL = 5000;

interface VideoSelectorProps {
  setVideoHash: (hash: string) => void;
}

export default function VideoSelector({ setVideoHash }: VideoSelectorProps) {
  const { userData } = useUser();
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [uploadedVideos, setUploadedVideos] = useState<Video[]>([]);
  const [videoProcessingStatuses, setVideoProcessingStatuses] = useState<
    Record<string, { status: string }>
  >({});
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const timeoutId = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    async function fetchUploadedVideos() {
      setIsLoading(true);
      try {
        const _uploadedVideos = await getUploadedVideos({
          user_email: userData.email,
        } as GetUploadedVideoParams);
        setUploadedVideos(_uploadedVideos);
      } catch (error) {
        console.error('Error fetching uploaded videos:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchUploadedVideos();
  }, [userData]);

  useEffect(() => {
    async function fetchVideoProcessingStatuses() {
      try {
        const data = await getVideoProcessingStatuses(userData.email);
        if (data.result && data.result !== 'error') {
          const newVideoProcessingStatuses = { ...videoProcessingStatuses };
          data.result.forEach(
            (result: {
              video_hash: string;
              status: string;
              error?: string;
            }) => {
              newVideoProcessingStatuses[result.video_hash] = {
                status: result.status,
              };
            }
          );
          setVideoProcessingStatuses(newVideoProcessingStatuses);
        }
      } catch (error) {
        console.error('Error fetching video processing statuses:', error);
      }
    }

    async function pollForDone() {
      try {
        const data = await getVideoProcessingStatuses(userData.email);
        let anyPending = false;
        if (data.result && data.result !== 'error') {
          let changed = false;
          const newVideoProcessingStatuses = { ...videoProcessingStatuses };
          const existingKeys = Object.keys(videoProcessingStatuses);
          data.result.forEach(
            (result: {
              video_hash: string;
              status: string;
              error?: string;
            }) => {
              const videoHash = result.video_hash;
              if (
                result.status === 'done' &&
                existingKeys.includes(videoHash)
              ) {
                delete newVideoProcessingStatuses[videoHash];
                changed = true;
              } else if (
                result.status === 'error' &&
                existingKeys.includes(videoHash) &&
                newVideoProcessingStatuses[videoHash].status !== 'error'
              ) {
                console.error(
                  `Error processing video ${videoHash}: ${result.error}`
                );
                newVideoProcessingStatuses[videoHash].status = 'error';
                changed = true;
              } else {
                anyPending = true;
              }
            }
          );
          if (changed) {
            setVideoProcessingStatuses(newVideoProcessingStatuses);
          }
        }
        if (anyPending) {
          timeoutId.current = setTimeout(pollForDone, POLL_INTERVAL);
        }
      } catch (error) {
        console.error('Error polling video processing statuses:', error);
      }
    }

    if (userData.email) {
      fetchVideoProcessingStatuses();
      pollForDone();
    }

    return () => {
      if (timeoutId.current) {
        clearTimeout(timeoutId.current);
      }
    };
  }, [userData.email, videoProcessingStatuses]);

  useEffect(() => {
    if (selectedVideo && selectedVideo.hash) {
      setVideoHash(selectedVideo.hash);
    }
  }, [selectedVideo, setVideoHash]);

  return (
    <div className="grid gap-3">
      {isLoading ? (
        <div className="grid items-center gap-1 m-auto">
          <div>Loading...</div>
          <LoadingSpinner size="large" />
        </div>
      ) : (
        <>
          <UploadVideo
            userEmail={userData.email}
            setVideoHash={setVideoHash}
            videoProcessingStatuses={videoProcessingStatuses}
            setVideoProcessingStatuses={setVideoProcessingStatuses}
          />
          <VideoTable
            uploadedVideos={uploadedVideos}
            selectedVideo={selectedVideo}
            videoProcessingStatuses={videoProcessingStatuses}
            setSelectedVideo={setSelectedVideo}
          />
        </>
      )}
    </div>
  );
}
