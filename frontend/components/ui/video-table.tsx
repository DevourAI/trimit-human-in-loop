import {
  CheckIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from '@radix-ui/react-icons';
import React from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { UploadedVideo } from '@/gen/openapi/api';
import { formatDuration } from '@/lib/utils';

interface VideoTableProps {
  uploadedVideos: UploadedVideo[];
  videoProcessingStatuses: { [key: string]: { status: string } };
  selectVideo: (video: UploadedVideo) => void;
  currentlySelectedVideoHash?: string;
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
    case 'uploading':
      return (
        <Badge variant="secondary">
          <ClockIcon className="mr-1" />
          Uploading
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
  currentlySelectedVideoHash,
}) => {
  return (
    <>
      <Label htmlFor="videoTable" className="cursor-pointer">
        Select an uploaded video
      </Label>
      <div className="rounded-md border">
        <Table id="videoTable">
          <TableHeader>
            <TableRow>
              <TableHead>Filename</TableHead>
              <TableHead className="hidden md:table-cell">Thumbnail</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Duration</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {uploadedVideos.length ? (
              uploadedVideos.map((video: UploadedVideo) => (
                <TableRow
                  key={video.filename}
                  data-state={
                    video.video_hash === currentlySelectedVideoHash &&
                    'selected'
                  }
                  onClick={() => selectVideo(video)}
                >
                  <TableCell className="font-medium">
                    {video.filename}
                  </TableCell>
                  <TableCell className="hidden md:table-cell">
                    <video width="320" height="240" controls>
                      <source src={video.remote_url} type="video/mp4" />
                      <track kind="captions" />
                      Your browser does not support the video tag.
                    </video>
                  </TableCell>
                  <TableCell className="font-medium">
                    {videoProcessingStatuses[video.video_hash]
                      ? getStatusBadge(
                          videoProcessingStatuses[video.video_hash].status
                        )
                      : getStatusBadge('done')}
                  </TableCell>
                  <TableCell className="font-medium">
                    {formatDuration(video.duration, { roundMs: true })}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="outline"
                      onClick={() => selectVideo(video)}
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
    </>
  );
};

export default VideoTable;
