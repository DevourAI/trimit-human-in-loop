import os

if os.getenv("DEPLOY_FRONTEND"):
    from trimit.frontend import *
#  if os.getenv("DEPLOY_BACKEND"):
#  from trimit.backend import *
