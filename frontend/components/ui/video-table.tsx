import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import type { Video } from '@/lib/types';

interface VideoTableProps {
  uploadedVideos: Video[];
  selectedVideo: Video | null;
  videoProcessingStatuses: { [key: string]: { status: string } };
  setSelectedVideo: (video: Video) => void;
}

const VideoTable: React.FC<VideoTableProps> = ({
  uploadedVideos,
  selectedVideo,
  videoProcessingStatuses,
  setSelectedVideo,
}) => {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Filename</TableHead>
            <TableHead>Thumbnail</TableHead>
            <TableHead>Processing Status</TableHead>
            <TableHead></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {uploadedVideos.length ? (
            uploadedVideos.map((video: Video) => (
              <TableRow
                key={video.filename}
                data-state={
                  video.filename == selectedVideo?.filename && 'selected'
                }
              >
                <TableCell className="font-medium">{video.filename}</TableCell>
                <TableCell>
                  <video width="320" height="240" controls>
                    <source src={video.remoteUrl} type="video/mp4" />
                    <track kind="captions" />
                    Your browser does not support the video tag.
                  </video>
                </TableCell>
                <TableCell className="font-medium">
                  {videoProcessingStatuses[video.hash]
                    ? videoProcessingStatuses[video.hash].status === 'error'
                      ? 'Error'
                      : videoProcessingStatuses[video.hash].status === 'pending'
                        ? 'Processing'
                        : 'Ready to use'
                    : 'Ready to use'}
                </TableCell>
                <TableCell>
                  <Button
                    variant="outline"
                    onClick={() => setSelectedVideo(video)}
                  >
                    Select
                  </Button>
                </TableCell>
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={999} className="h-24 text-center">
                No videos uploaded.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
};

export default VideoTable;
