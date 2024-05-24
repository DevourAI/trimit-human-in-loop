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


export default function VideoSelector({userData, setVideoHash}) {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [uploadedVideos, setUploadedVideos] = useState([]);

  useEffect(() => {
    async function fetchUploadedVideos() {
        const _uploadedVideos = await getUploadedVideos({user_email: userData.email} as GetUploadedVideoParams);
        console.log('In VideoSelector: uploadedVideos', _uploadedVideos);
        setUploadedVideos(_uploadedVideos);
        setSelectedVideo(_uploadedVideos[0]);
    }

    fetchUploadedVideos();
  }, [userData]);

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
                </TableRow>
              )) : (
                <TableRow>
                  <TableCell className="h-24 text-center">
                    No results.
                  </TableCell>
                </TableRow>

              )
            }
        </TableBody>
      </Table>
    </div>
  )
}
