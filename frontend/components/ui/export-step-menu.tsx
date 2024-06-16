import * as React from 'react';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { DownloadIcon } from '@radix-ui/react-icons';
import {
  downloadVideo,
  downloadTimeline,
  downloadTranscriptText,
  downloadSoundbitesText,
} from '@/lib/api';
import { DownloadFileParams } from '@/lib/types';

interface DownloadButtonsProps {
  userParams: DownloadFileParams;
  stepName?: string;
  substepName?: string;
}

export default function ExportStepMenu({
  userParams,
  stepName,
  substepName,
}: DownloadButtonsProps) {
  const downloadParams: DownloadFileParams = {
    ...userParams,
    step_name: stepName,
    substep_name: substepName,
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button className="w-fit">
          <DownloadIcon className="mr-2" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuItem onSelect={() => downloadVideo(downloadParams)}>
          Download video
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => downloadTimeline(downloadParams)}>
          Download timeline
        </DropdownMenuItem>
        <DropdownMenuItem
          onSelect={() => downloadTranscriptText(downloadParams)}
        >
          Download transcript
        </DropdownMenuItem>
        <DropdownMenuItem
          onSelect={() => downloadSoundbitesText(downloadParams)}
        >
          Download soundbites transcript
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
