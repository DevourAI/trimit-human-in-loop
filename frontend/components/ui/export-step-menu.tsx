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
  const fileKeysToNames = {
    video: 'Video',
    video_timeline: 'Video Timeline',
    transcript_text: 'Transcript Text',
    soundbites_text: 'Soundbites Text',
    soundbites_timeline: 'Soundbites Timeline',
  };
  const loadingStatus = Object.keys(fileKeysToNames).reduce(
    (acc, key) => {
      acc[key] = !exportResult || exportResult[key] === null;
      return acc;
    },
    {} as Record<string, boolean>
  );
  const foundStatus = Object.keys(fileKeysToNames).reduce(
    (acc, key) => {
      acc[key] =
        (!loadingStatus[key] &&
          exportResult &&
          exportResult[key] !== null &&
          exportResult[key] !== undefined) ||
        false;
      return acc;
    },
    {} as Record<string, boolean>
  );
  const downloadFns = {
    video: downloadVideo,
    video_timeline: downloadTimeline,
    transcript_text: downloadTranscriptText,
    soundbites_text: downloadSoundbitesText,
    soundbites_timeline: downloadSoundbitesTimeline,
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
        {Object.entries(fileKeysToNames).map(([filekey, name]) => {
          const loading = loadingStatus[filekey];
          const found = foundStatus[filekey];
          const downloadFn = downloadFns[filekey as keyof typeof downloadFns];
          if (!loading && !found) return null;
          return (
            <DropdownMenuItem
              key={filekey}
              disabled={disabled || !found}
              onSelect={() => downloadFn(downloadParams)}
            >
              {loading ? <LoadingSpinner size="small" /> : null}
              Download {name}
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
