import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { ReloadIcon } from "@radix-ui/react-icons"
import React, { useState } from 'react';


export default function UploadVideo({uploadVideo}) {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedVideo(file);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (selectedVideo) {
      setIsUploading(true)
      try {
        await uploadVideo(selectedVideo)
      } catch (error) {
        console.error('Error uploading video', error)
      }
      setIsUploading(false)
    }
  };

  return (
    <div className="grid w-full max-w-sm items-center gap-1.5">
      <form onSubmit={handleSubmit}>
        <Input id="video" onChange={handleVideoChange} accept="video/*" type="file" />
        { isUploading ?
          <Button disabled type="submit">
            <ReloadIcon className="mr-2 h-4 w-4 animate-spin" />
            Uploading...
          </Button>
          :
          <Button type="submit">Upload Video</Button>
        }
      </form>
      {selectedVideo && (
        <div>
          <h4>Selected Video:</h4>
          <p>{selectedVideo.name}</p>
          <video width="320" height="240" controls>
            <source src={URL.createObjectURL(selectedVideo)} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      )}
    </div>
  )
}
