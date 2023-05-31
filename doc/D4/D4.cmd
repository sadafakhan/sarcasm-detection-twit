executable = src/adaptation_task.sh
getenv     = true
output     = D4.out
error      = D4.error
log        = D4.log
arguments  = "setup/D4.yml"
notification = complete
transfer_executable = false
request_memory = 2*1024
queue
