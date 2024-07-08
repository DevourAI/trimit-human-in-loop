import { DownloadIcon } from '@radix-ui/react-icons';
import * as React from 'react';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  downloadSoundbitesText,
  downloadTimeline,
  downloadTranscriptText,
  downloadVideo,
} from '@/lib/api';
import { DownloadFileParams } from '@/lib/types';

interface DownloadButtonsProps {
  userParams: DownloadFileParams;
  stepName?: string;
  disabled: boolean;
}

export default function ExportStepMenu({
  userParams,
  stepName,
  disabled,
}: DownloadButtonsProps) {
  const downloadParams: DownloadFileParams = {
    ...userParams,
    step_name: stepName,
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          disabled={disabled}
          className="w-fit"
          variant="secondary"
          size="sm"
        >
          <DownloadIcon className="mr-2" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuItem
          disabled={disabled}
          onSelect={() => downloadVideo(downloadParams)}
        >
          Download video
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={disabled}
          onSelect={() => downloadTimeline(downloadParams)}
        >
          Download timeline
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={disabled}
          onSelect={() => downloadTranscriptText(downloadParams)}
        >
          Download transcript
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={disabled}
          onSelect={() => downloadSoundbitesText(downloadParams)}
        >
          Download soundbites transcript
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
