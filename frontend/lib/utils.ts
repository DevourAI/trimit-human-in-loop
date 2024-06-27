import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const removeEmptyVals = (d) => {
  return Object.keys(d).reduce((acc, key) => {
    if (d[key] !== undefined && d[key] !== null && d[key] !== '') {
      acc[key] = d[key];
    }
    return acc;
  }, {});
};
