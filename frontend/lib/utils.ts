import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const removeEmptyVals = (d: Record<string, any>) => {
  return Object.keys(d).reduce(
    (acc, key) => {
      if (d[key] !== undefined && d[key] !== null && d[key] !== '') {
        acc[key] = d[key];
      }
      return acc;
    },
    {} as Record<string, any>
  );
};

export function formatDuration(
  duration: number,
  options: { roundMs: boolean }
): string {
  // Calculate hours, minutes, seconds, and milliseconds
  const hours = Math.floor(duration / 3600);
  const minutes = Math.floor((duration % 3600) / 60);
  let seconds = Math.floor(duration % 60);
  let milliseconds = Math.floor((duration % 1) * 1000);
  if (options.roundMs) {
    seconds += Math.round(milliseconds / 1000);
    milliseconds = 0;
  }

  // Format as HH:MM:SS:MS or HH:MM:SS
  const formattedHours = String(hours).padStart(2, '0');
  const formattedMinutes = String(minutes).padStart(2, '0');
  const formattedSeconds = String(seconds).padStart(2, '0');
  const formattedMilliseconds = String(milliseconds).padStart(3, '0');
  let milliSuffix = options.roundMs ? '' : `.${formattedMilliseconds}`;

  return `${formattedHours}:${formattedMinutes}:${formattedSeconds}${milliSuffix}`;
}
