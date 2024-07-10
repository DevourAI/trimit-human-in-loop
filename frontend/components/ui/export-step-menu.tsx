import { DownloadIcon } from '@radix-ui/react-icons';
import * as React from 'react';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { useStructuredInputForm } from '@/contexts/structured-input-form-context';
import {
  downloadSoundbitesText,
  downloadSoundbitesTimeline,
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

  const { exportResult } = useStructuredInputForm();
  const videoLoading = !exportResult || exportResult.video === null;
  const videoFound =
    !videoLoading &&
    exportResult.video !== null &&
    exportResult.video !== undefined;

  const timelineLoading = !exportResult || exportResult.video_timeline === null;
  const timelineFound =
    !timelineLoading &&
    exportResult.timeline !== null &&
    exportResult.timeline !== undefined;

  const transcriptTextLoading =
    !exportResult || exportResult.transcript_text === null;
  const transcriptTextFound =
    !transcriptTextLoading &&
    exportResult.transcript_text !== null &&
    exportResult.transcript_text !== undefined;

  const soundbitesTextLoading =
    !exportResult || exportResult.soundbites_text === null;
  const soundbitesTextFound =
    !soundbitesTextLoading &&
    exportResult.soundbites_text !== null &&
    exportResult.soundbites_text !== undefined;

  const soundbitesTimelineLoading =
    !exportResult || exportResult.soundbites_timeline === null;
  const soundbitesTimelineFound =
    !soundbitesTimelineLoading &&
    exportResult.soundbites_timeline !== null &&
    exportResult.soundbites_timeline !== undefined;

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
        {videoLoading || videoFound ? (
          <DropdownMenuItem
            disabled={disabled || !videoFound}
            onSelect={() => downloadVideo(downloadParams)}
          >
            {videoLoading ? <LoadingSpinner size="small" /> : null}
            Download video
          </DropdownMenuItem>
        ) : null}
        {timelineLoading || timelineFound ? (
          <DropdownMenuItem
            disabled={disabled || !timelineFound}
            onSelect={() => downloadTimeline(downloadParams)}
          >
            {timelineLoading ? <LoadingSpinner size="small" /> : null}
            Download timeline
          </DropdownMenuItem>
        ) : null}
        {transcriptTextLoading || transcriptTextFound ? (
          <DropdownMenuItem
            disabled={disabled || !transcriptTextFound}
            onSelect={() => downloadTranscriptText(downloadParams)}
          >
            {transcriptTextLoading ? <LoadingSpinner size="small" /> : null}
            Download transcript
          </DropdownMenuItem>
        ) : null}
        {soundbitesTextLoading || soundbitesTextFound ? (
          <DropdownMenuItem
            disabled={disabled || !soundbitesTextFound}
            onSelect={() => downloadSoundbitesText(downloadParams)}
          >
            {soundbitesTextLoading ? <LoadingSpinner size="small" /> : null}
            Download soundbites transcript
          </DropdownMenuItem>
        ) : null}
        {soundbitesTimelineLoading || soundbitesTimelineFound ? (
          <DropdownMenuItem
            disabled={disabled || !soundbitesTimelineFound}
            onSelect={() => downloadSoundbitesTimeline(downloadParams)}
          >
            {!soundbitesTimelineFound ? <LoadingSpinner size="small" /> : null}
            Download soundbites timeline
          </DropdownMenuItem>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
