executable = src/primary_task.sh
getenv     = true
output     = D2.out
error      = D2.error
log        = D2.log
arguments   = "setup/D2.yml"
notification = complete
transfer_executable = false
request_memory = 2*1024
queue
