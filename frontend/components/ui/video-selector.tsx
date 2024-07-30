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
  selectedVideo: UploadedVideo | null;
  setSelectedVideo: (video: UploadedVideo) => void;
}

interface UploadingVideo {
  filename: string;
  title: string;
}

export default function VideoSelector({
  selectedVideo,
  setSelectedVideo,
}: VideoSelectorProps) {
  const { userData } = useUser();
  const [uploadedVideos, setUploadedVideos] = useState<UploadedVideo[]>([]);
  const [uploadingVideo, setUploadingVideo] = useState<UploadingVideo | null>(
    null
  );
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
    if (!userData.email) return;

    const fetchUploadedVideos = async () => {
      //setIsLoading(true);
      try {
        const videos = await getUploadedVideos({
          user_email: userData.email,
        } as GetUploadedVideoParams);

        if (uploadingVideo) {
          if (
            videos.filter(
              (video) =>
                video.filename_or_weblink === uploadingVideo.filename_or_weblink
            ).length === 0
          ) {
            videos.push({
              title: uploadingVideo.title,
              filename: uploadingVideo.filename_or_weblink,
              video_hash: 'tmp',
            } as UploadedVideo);
          }
        }

        return videos as UploadedVideo[];
      } catch (error) {
        console.error('Error fetching uploaded videos:', error);
      } finally {
        //setIsLoading(false);
      }
      return [];
    };

    const fetchVideoProcessingStatuses = async () => {
      if (isPolling.current) return; // Ensure only one polling request in flight
      isPolling.current = true;
      const newUploadedVideos = await fetchUploadedVideos();
      if (newUploadedVideos.length > 0 && selectedVideo === null) {
        setSelectedVideo(newUploadedVideos[0]);
      }
      setUploadedVideos(newUploadedVideos);
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
              const filename = newUploadedVideos.find(
                (video) => video.video_hash === result.video_hash
              )?.filename;
              acc[result.video_hash] = {
                status:
                  filename === uploadingVideo ? 'uploading' : result.status,
              };
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
  }, [userData.email, uploadingVideo, selectedVideo, setSelectedVideo]);

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
                setUploadingVideo={setUploadingVideo}
                setVideoDetails={(
                  hash: string | null,
                  filename: string,
                  title: string
                ) => {
                  setSelectedVideo({
                    video_hash: hash || '',
                    filename,
                    title,
                    path: '',
                    duration: 0,
                    remote_url: '',
                  });
                  if (
                    uploadedVideos.filter(
                      (video) => video.filename === filename
                    ).length === 0
                  ) {
                    setUploadedVideos([
                      ...uploadedVideos,
                      { filename, video_hash: 'tmp' } as UploadedVideo,
                    ]);
                    setVideoProcessingStatuses({
                      ...videoProcessingStatuses,
                      tmp: { status: 'uploading' },
                    });
                  } else if (!hash) {
                    const hash = uploadedVideos.find(
                      (video) => video.filename === filename
                    )?.video_hash;
                    if (hash) {
                      setVideoProcessingStatuses({
                        ...videoProcessingStatuses,
                        [hash]: { status: 'uploading' },
                      });
                    }
                  } else {
                    setVideoProcessingStatuses({
                      ...videoProcessingStatuses,
                      [hash]: { status: 'pending' },
                    });
                  }
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
