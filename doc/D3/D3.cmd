executable = src/primary_task.sh
getenv     = true
output     = D3.out
error      = D3.error
log        = D3.log
arguments   = "setup/D3.yml"
notification = complete
transfer_executable = false
request_memory = 2*1024
queue
