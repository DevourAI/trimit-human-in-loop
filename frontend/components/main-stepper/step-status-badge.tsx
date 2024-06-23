import { CheckIcon, Cross1Icon } from '@radix-ui/react-icons';
import { FC } from 'react';

import { Badge } from '@/components/ui/badge';

const StepStatusBadge: FC<{ done: boolean }> = ({ done }) => (
  <Badge variant={done ? 'default' : 'destructive'}>
    {done ? (
      <>
        <CheckIcon className="mr-1 h-3 w-3" /> Complete
      </>
    ) : (
      <>
        <Cross1Icon className="mr-1 h-3 w-3" /> Incomplete
      </>
    )}
  </Badge>
);

export default StepStatusBadge;
