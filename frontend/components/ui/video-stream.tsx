import { FC } from 'react';

import { remoteVideoStreamURLForPath } from '@/lib/api';

export const VideoStream: FC<{ videoPath: string }> = ({ videoPath }) => {
  const remoteUrl = remoteVideoStreamURLForPath(videoPath);
  return (
    <video width="320" height="240" controls>
      <source src={remoteUrl} type="video/mp4" />
      <track kind="captions" />
      Your browser does not support the video tag.
    </video>
  );
};
