import React, { useEffect, useRef, useState } from 'react';

import { Card, CardContent } from '@/components/ui/card';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import UploadVideo from '@/components/ui/upload-video';
import VideoTable from '@/components/ui/video-table';
import { useUser } from '@/contexts/user-context';
import { UploadedVideo } from '@/gen/openapi/api';
import { getUploadedVideos, getVideoProcessingStatuses } from '@/lib/api';
import type { GetUploadedVideoParams } from '@/lib/types';

const POLL_INTERVAL = 5000;

interface VideoSelectorProps {
  setVideoDetails: (hash: string, filename: string) => void;
}

export default function VideoSelector({ setVideoDetails }: VideoSelectorProps) {
  const { userData } = useUser();
  const [selectedVideo, setSelectedVideo] = useState<UploadedVideo | null>(
    null
  );
  const [uploadedVideos, setUploadedVideos] = useState<UploadedVideo[]>([]);
  const [videoProcessingStatuses, setVideoProcessingStatuses] = useState<
    Record<string, { status: string }>
  >({});
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const timeoutId = useRef<NodeJS.Timeout | null>(null);
  const isComponentMounted = useRef<boolean>(true);
  const isPolling = useRef<boolean>(false);

  // Used to cancel the timeout when the component unmounts
  useEffect(() => {
    isComponentMounted.current = true;
    return () => {
      isComponentMounted.current = false;
    };
  }, []);

  useEffect(() => {
    const fetchUploadedVideos = async () => {
      setIsLoading(true);
      try {
        const videos = await getUploadedVideos({
          user_email: userData.email,
        } as GetUploadedVideoParams);

        setUploadedVideos(videos as UploadedVideo[]);
        if (videos.length > 0) setSelectedVideo(videos[0]);
      } catch (error) {
        console.error('Error fetching uploaded videos:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUploadedVideos();
  }, [userData]);

  useEffect(() => {
    if (!userData.email) return;

    const fetchVideoProcessingStatuses = async () => {
      if (isPolling.current) return; // Ensure only one polling request in flight
      isPolling.current = true;
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
      } finally {
        isPolling.current = false;
      }
    };

    const pollForStatuses = async () => {
      await fetchVideoProcessingStatuses();
      if (isComponentMounted.current) {
        timeoutId.current = setTimeout(pollForStatuses, POLL_INTERVAL);
      }
    };

    pollForStatuses();

    return () => {
      if (timeoutId.current) {
        clearTimeout(timeoutId.current);
      }
    };
  }, [userData.email]);

  useEffect(() => {
    if (selectedVideo?.video_hash) {
      setVideoDetails(selectedVideo.video_hash, selectedVideo.filename);
    }
  }, [selectedVideo, setVideoDetails]);

  return (
    <div className="grid gap-3">
      {isLoading ? (
        <div className="grid items-center gap-1 m-auto">
          <div>Loading...</div>
          <LoadingSpinner size="large" />
        </div>
      ) : (
        <Card className="max-w-full shadow-none">
          <CardContent className="flex max-w-full p-0 relative">
            <div className="w-1/2 p-4">
              <UploadVideo
                userEmail={userData.email}
                setVideoDetails={(hash: string, filename: string) => {
                  setVideoDetails(hash, filename);
                }}
                videoProcessingStatuses={videoProcessingStatuses}
                setVideoProcessingStatuses={setVideoProcessingStatuses}
              />
            </div>
            <div className="w-1/2 p-8 border-l">
              <VideoTable
                uploadedVideos={uploadedVideos}
                videoProcessingStatuses={videoProcessingStatuses}
                selectVideo={setSelectedVideo}
                currentlySelectedVideoHash={selectedVideo?.video_hash}
              />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
