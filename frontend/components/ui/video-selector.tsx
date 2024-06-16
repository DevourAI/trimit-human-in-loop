import React, { useEffect, useRef, useState } from 'react';

import { LoadingSpinner } from '@/components/ui/loading-spinner';
import UploadVideo from '@/components/ui/upload-video';
import VideoTable from '@/components/ui/video-table';
import { useUser } from '@/contexts/user-context';
import { getUploadedVideos, getVideoProcessingStatuses } from '@/lib/api';
import type { GetUploadedVideoParams, Video } from '@/lib/types';

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
    const fetchUploadedVideos = async () => {
      setIsLoading(true);
      try {
        const videos = await getUploadedVideos({
          user_email: userData.email,
        } as GetUploadedVideoParams);

        setUploadedVideos(videos as Video[]);
      } catch (error) {
        console.error('Error fetching uploaded videos:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUploadedVideos();
  }, [userData]);

  useEffect(() => {
    const fetchVideoProcessingStatuses = async () => {
      try {
        const data = await getVideoProcessingStatuses(userData.email);
        if (
          data &&
          typeof data === 'object' &&
          'result' in data &&
          data.result !== 'error'
        ) {
          const newStatuses = (
            data.result as { video_hash: string; status: string }[]
          ).reduce(
            (
              acc: Record<string, { status: string }>,
              result: { video_hash: string; status: string }
            ) => {
              acc[result.video_hash] = { status: result.status };
              return acc;
            },
            {}
          );
          setVideoProcessingStatuses(newStatuses);
        }
      } catch (error) {
        console.error('Error fetching video processing statuses:', error);
      }
    };

    const pollForStatuses = async () => {
      await fetchVideoProcessingStatuses();
      timeoutId.current = setTimeout(pollForStatuses, POLL_INTERVAL);
    };

    if (userData.email) {
      fetchVideoProcessingStatuses();
      pollForStatuses();
    }

    return () => {
      if (timeoutId.current) {
        clearTimeout(timeoutId.current);
      }
    };
  }, [userData.email]);

  useEffect(() => {
    if (selectedVideo?.hash) {
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
            videoProcessingStatuses={videoProcessingStatuses}
            selectVideo={setSelectedVideo}
          />
        </>
      )}
    </div>
  );
}
