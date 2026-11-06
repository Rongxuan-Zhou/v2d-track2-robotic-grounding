# 仓根入口。远端执行见 scripts/dev/remote 的注释。
.PHONY: sync remote shell pull-results

sync:                      ## 同步工作树到训练机，不执行
	@scripts/dev/remote

remote:                    ## 同步并在训练机执行：make remote CMD="python ..."
	@scripts/dev/remote $(CMD)

shell:                     ## 登录训练机的仓库目录
	@. ./.env && ssh -t $$V2D_HOST "cd $$V2D_REMOTE && exec bash -l"

init-remote:               ## 一次性准备远端(建目录+哨兵+推 .env)
	@scripts/dev/init-remote

pull-results:              ## 回流 results/
	@scripts/dev/pull-results
