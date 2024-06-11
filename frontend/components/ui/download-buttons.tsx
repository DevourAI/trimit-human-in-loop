import { Button } from "@/components/ui/button"
import {
  downloadVideo,
  downloadTimeline,
  downloadTranscriptText,
  downloadSoundbitesText,
  getVideoProcessingStatuses,
} from "@/lib/api";

export default function DownloadButtons({userParams, ...options}) {
   const downloadParams = {
      ...userParams,
   }
   if (options.stepName) {
      downloadParams.step_name = options.stepName
   }
   if (options.substepName) {
      downloadParams.substep_name = options.substepName
   }
   console.log('download params', downloadParams)

  return (
    <div className="grid w-full max-w-sm items-center gap-1.5">
       <Button onClick={() => downloadVideo(downloadParams)}>
          Download video
       </Button>
       <Button onClick={() => downloadTimeline(downloadParams)}>
          Download timeline
       </Button>
       <Button onClick={() => downloadTranscriptText(downloadParams)}>
          Download transcript
       </Button>
       <Button onClick={() => downloadSoundbitesText(downloadParams)}>
          Download soundbites transcript
       </Button>
    </div>
  )
}
