import React, { useEffect, useState } from 'react';
import { type GetUploadedVideoParams, Video } from "@/lib/types";
import { getUploadedVideos } from "@/lib/api";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export default function VideoSelector({userData, videoProcessingStatuses, setVideoHash}) {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [uploadedVideos, setUploadedVideos] = useState([]);

  useEffect(() => {
    async function fetchUploadedVideos() {
        const _uploadedVideos = await getUploadedVideos({user_email: userData.email} as GetUploadedVideoParams);
        setUploadedVideos(_uploadedVideos);
        setSelectedVideo(_uploadedVideos[0]);
    }

    fetchUploadedVideos();
  }, [userData, videoProcessingStatuses]);

  useEffect(() => {
    if (selectedVideo && selectedVideo.hash) {
      setVideoHash(selectedVideo.hash);
    }
  }, [selectedVideo]);

  return (
    <div className="rounded-md border">
      <Table>
        <TableCaption>Your uploaded videos</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Filename</TableHead>
            <TableHead>Thumbnail</TableHead>
            <TableHead>Processing Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
            {uploadedVideos.length ?
              uploadedVideos.map((video) => (
                <TableRow onClick={() => setSelectedVideo(video)} key={video.filename} data-state={video.filename == selectedVideo.filename && "selected"}>
                  <TableCell className="font-medium">{video.filename}</TableCell>
                  <TableCell>
                    <video width="320" height="240" controls>
                      <source src={video.remoteUrl} type="video/mp4" />
                      Your browser does not support the video tag.
                    </video>
                  </TableCell>
                  <TableCell className="font-medium">{
                    videoProcessingStatuses[video.hash]?
                      videoProcessingStatuses[video.hash].status === "error" ?
                        "Error"
                      : videoProcessingStatuses[video.hash].status === "pending" ?
                        "Processing"
                      : "Ready to use"
                    : "Ready to use"
                  }</TableCell>
                </TableRow>
              )) : (
                <TableRow>
                  <TableCell className="h-24 text-center">
                    No results.
                  </TableCell>
                  <TableCell className="h-24 text-center">
                  </TableCell>
                  <TableCell className="h-24 text-center">
                  </TableCell>
                </TableRow>

              )
            }
        </TableBody>
      </Table>
    </div>
  )
}
