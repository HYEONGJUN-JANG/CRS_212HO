from loguru import logger

from CRS_212HO.evaluate_conv import ConvEvaluator


def train_conversation(args, model, train_dataloader, test_dataloader, path, results_file_path):
    # if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
    #     self.model.freeze_parameters()
    # else:
    #     self.model.module.freeze_parameters()
    # self.init_optim(self.conv_optim_opt, self.model.parameters())

    for epoch in range(args.conv_epoch):
        logger.info(f'[Conversation epoch {str(epoch)}]')
        logger.info('[Train]')
        for batch in train_dataloader.get_conv_data(batch_size=args.conv_batch_size, shuffle=False):
            context_tokens, response = batch
            scores_ft = model.forward(context_entities, context_tokens)
            loss_ft = model.criterion(scores_ft, target_items.to(args.device_id))

            if 'none' not in args.name:
                loss_pt = model.pre_forward(plot_meta, plot, plot_mask, review_meta, review, review_mask, target_items)
                loss = loss_ft + (loss_pt * args.loss_lambda)
            else:
                loss = loss_ft

            total_loss += loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    # test
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
