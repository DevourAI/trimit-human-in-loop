import {
  CheckIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from '@radix-ui/react-icons';
import React from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { Video } from '@/lib/types';

interface VideoTableProps {
  uploadedVideos: Video[];
  videoProcessingStatuses: { [key: string]: { status: string } };
  selectVideo: (video: Video) => void;
}

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'error':
      return (
        <Badge variant="destructive">
          <ExclamationTriangleIcon className="mr-1" />
          Error
        </Badge>
      );
    case 'pending':
      return (
        <Badge variant="secondary">
          <ClockIcon className="mr-1" />
          Processing
        </Badge>
      );
    case 'done':
    default:
      return (
        <Badge variant="secondary">
          <CheckIcon className="mr-1" />
          Ready
        </Badge>
      );
  }
};

const VideoTable: React.FC<VideoTableProps> = ({
  uploadedVideos,
  videoProcessingStatuses,
  selectVideo,
}) => {
  console.log('uploadedVideos', uploadedVideos);
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Filename</TableHead>
            <TableHead className="hidden md:table-cell">Thumbnail</TableHead>
            <TableHead>Status</TableHead>
            <TableHead></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {uploadedVideos.length ? (
            uploadedVideos.map((video: Video) => (
              <TableRow key={video.filename}>
                <TableCell className="font-medium">{video.filename}</TableCell>
                <TableCell className="hidden md:table-cell">
                  <video width="320" height="240" controls>
                    <source src={video.remote_url} type="video/mp4" />
                    <track kind="captions" />
                    Your browser does not support the video tag.
                  </video>
                </TableCell>
                <TableCell className="font-medium">
                  {videoProcessingStatuses[video.hash]
                    ? getStatusBadge(videoProcessingStatuses[video.hash].status)
                    : getStatusBadge('done')}
                </TableCell>
                <TableCell>
                  <Button variant="outline" onClick={() => selectVideo(video)}>
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
