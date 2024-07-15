import React, { FC } from 'react';

import { VideoStream } from '@/components/ui/video-stream';
import { OutputComponentProps } from '@/lib/types';

const Transcript: FC<{ text: string }> = ({ text }) => {
  return <p>{text}</p>;
};

export const VideoOutput: FC<OutputComponentProps> = ({
  value,
  exportResult,
}) => {
  const videoPath = exportResult?.video || '';
  return (
    <div className="relative p-3">
      <VideoStream videoPath={videoPath} />
      <Transcript text={value} />
    </div>
  );
};
